import json, os
from litellm import completion, completion_cost
from typing import Dict, List, Any
import re, traceback, ast, time
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from crm_sandbox.agents.prompts import SCHEMA_STRING, SYSTEM_METADATA, NATIVE_FC_PROMPT, CUSTOM_FC_PROMPT, FC_RULE_STRING, FC_FLEX_PROMPT
from crm_sandbox.agents.utils import parse_wrapped_response, BEDROCK_MODELS_MAP, TOGETHER_MODELS_MAP, VERTEX_MODELS_MAP, fc_prompt_builder


@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(10))
def chat_completion_request(
    messages,
    model,
    tools=None,
    temperature: float = 0.0,
    top_p=1.0,
    max_tokens=3500,
    additional_drop_params=[]
):
    res = completion(
        messages=messages,
        model=model,
        temperature=0.0,
        top_p=1.0,
        max_tokens=3500 if model not in ["o1-mini", "o1-preview", "o1-2024-12-17", "deepseek-r1", "o3-mini-2025-01-31", "gemini-2.5-flash-preview-04-17", "gemini-2.5-flash-preview-04-17-thinking-4096", "gemini-2.5-pro-preview-03-25"] else 50000,
        tools=tools if "llama" not in model else None, ## llama tool_calling through prompt
        additional_drop_params=["temperature", "top_p"] if model in ["o1-mini", "o1-preview", "o1-2024-12-17"] else []
    )
    return res
    
class ToolCallAgent:
    def __init__(
        self, tools, schema_obj, model: str = "gpt-4o", max_turns: int = 20, eval_mode="default", strategy="tool_call", provider="bedrock"
    ):
        schema = self._build_schema(schema_obj)
        self.tools = tools
        
        if strategy == "tool_call":
            if "llama" in model: # llama tool_calling through prompt
                self.sys_prompt = CUSTOM_FC_PROMPT.format(native_fc_prompt=NATIVE_FC_PROMPT.format(system="Salesforce instance"), tools_prompt=fc_prompt_builder(tools))
            else:
                self.sys_prompt = NATIVE_FC_PROMPT.format(system="Salesforce instance")
        else:
            self.sys_prompt = FC_FLEX_PROMPT.format(system_description=schema, system="Salesforce instance")
            
        self.model = model
        self.eval_mode = eval_mode
        self.max_turns = max_turns
        self.usage = {"cost": [], "completion_tokens": [], "prompt_tokens": [], "total_tokens": []}
        self.provider = provider
        if provider == "bedrock" and self.model in BEDROCK_MODELS_MAP:
            os.environ["AWS_REGION_NAME"] = BEDROCK_MODELS_MAP[self.model]["region"]
            self.model = BEDROCK_MODELS_MAP[self.model]["name"]
        elif provider == "together_ai" and self.model in TOGETHER_MODELS_MAP:
            self.model = TOGETHER_MODELS_MAP[self.model]["name"]
        elif "vertex" in provider and self.model in VERTEX_MODELS_MAP:
            self.model = VERTEX_MODELS_MAP[self.model]["name"]
        else:
            assert self.model in ["o1-mini", "o1-2024-12-17", "o1-preview", "gpt-4o-2024-08-06", "gpt-3.5-turbo-0125"], "Invalid model name"
            

    def _build_schema(self, schema_obj):
        object_description = dict()
        for item in schema_obj:
            object_description[item["object"]] = "\n".join([f"  - {k}: {v}" for k,v in item["fields"].items()])
            
        template = SCHEMA_STRING.format(
            object_names=", ".join(object_description.keys()),
            object_fields="\n".join(
                [f"{obj}\n{fields}" for obj, fields in object_description.items()]
            )
        )
        return template
    
    def reset(self, args):
        if args["metadata"]["required"]:
            self.sys_prompt += SYSTEM_METADATA.format(system_metadata=args["metadata"]["required"], system="Salesforce instance") # add task/query-specific metadata here
        if self.eval_mode == "aided" and "optional" in args["metadata"]:
            self.sys_prompt += "\n" + args["metadata"]["optional"]
        if self.model not in ["o1-mini", "o1-2024-12-17", "o1-preview"]:
            self.messages = [{"role": "system", "content": self.sys_prompt.strip()}]
            self.messages.append({"role": "user", "content": args["query"].strip()})
        else:
            # No system role for o1-mini and o1-preview
            self.messages = [{"role": "user", "content": self.sys_prompt + "\n\n" + args["query"]}]
        self.usage = {"cost": [], "completion_tokens": [], "prompt_tokens": [], "total_tokens": []}
        
    def act(self, env, index=None, temperature=0.0):
        query, metadata = env.reset(task_index=index)
        self.reset({"query": query, "metadata": metadata})
        self.info = {}
        self.info["observation_sizes"] = []
        done = False
        reward = 0
        
        for turn_id in range(self.max_turns):
            time.sleep(3)
            info = {}
            res = chat_completion_request(
                messages=self.messages,
                model=self.model,
                temperature=0.0,
                top_p=1.0,
                max_tokens=3500,
                tools=self.tools if "llama" not in self.model else None, ## llama tool_calling through prompt
                additional_drop_params=["temperature"] if self.model in ["o1-mini", "o1-preview", "o1-2024-12-17"] else []
            )
            message = res.choices[0].message.model_dump()
            usage = res.usage
            
            for key in self.usage.keys():
                if key != "cost":
                    self.usage[key].append(usage.get(key, 0))
            self.usage["cost"].append(res._hidden_params["response_cost"])
            
            print("message", message, flush=True)
            action = self.message_action_parser(message)
            print("#", turn_id, "Agent action:", action, flush=True)
            
            
            if action is None:
                self.info["end_reason"] = {
                    "source": "agent",
                    "message": "Invalid action",
                    "content":  message["content"].strip()
                }
                info["end_reason"] = self.info["end_reason"]
                # if tool_call attempted but failed
                if "llama" not in self.model and "tool_calls" in message and message["tool_calls"] is not None and len(message["tool_calls"]) > 0 and "id" in message["tool_calls"][0]:
                    message["tool_calls"] = message["tool_calls"][:1]
                    self.messages.append(message)
                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": message["tool_calls"][0]["id"].strip(),
                            "name": message["tool_calls"][0]["function"]["name"].strip(),
                            "content": f"Invalid tool call argument. Please make a valid tool call using the tools provided or submit the final answer using the 'respond' tool."
                        }
                    )
                # if no valid tool_call or not llama
                else:
                    if "llama" in self.model:
                        self.messages.append({"role": "assistant",
                                              "content": message["content"].strip()
                                            })
                    else:
                        self.messages.append(message)
                    self.messages.append({"role": "user", "content": FC_RULE_STRING})
                continue
            else:
                if "llama" in self.model:
                    message = {
                        "role": "assistant",
                        "content": f"Action: {action['name']}\nAction Input: {json.dumps(action['arguments'])}"
                    }
                    self.messages.append(message)
                else:
                    message["tool_calls"] = message["tool_calls"][:1]
                    self.messages.append(message)
            
            obs, reward, done, info = env.step(action)
            if "observation_size" in info:
                self.info["observation_sizes"].append(info["observation_size"])
            if "end_reason" in info: # implies error in query
                self.info["end_reason"] = info["end_reason"]
            if done: # implies submit action
                break
            else:
                if "llama" in self.model:
                    self.messages.append({"role": "user", "content": f"Salesforce instance output: {obs}"})
                else:
                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": message["tool_calls"][0]["id"].strip(),
                            "name": action["name"],
                            "content": obs
                        }
                    )
        
        # Here when either max_turns is reached or submitted
        if not done: 
            if "end_reason" not in info: # no error in last query
                self.info["end_reason"] = {
                    "source": "agent",
                    "message": "Max turns reached",
                    "content":  message["content"].strip()
                }
        self.info["usage"] = self.usage
        self.info["total_cost"] = sum(self.usage["cost"])
        self.info["num_turns"] = turn_id + 1
        return reward

    def get_messages(self) -> List[Dict[str, str]]:
        return self.messages

    def message_action_parser(self, message: str) -> Dict[str, str]:
        if "llama" in self.model:
            parsed_output = self.base_action_parser(message["content"].strip())
            action = {}
            # no tool call and non-null message
            if len(parsed_output) == 1 and parsed_output[0]:
                action["name"] = "respond"
                action["arguments"] = parsed_output[0]
                return action
            # proper tool call
            if len(parsed_output) == 2 and parsed_output[0] and parsed_output[1]:
                action["name"], action["arguments"] = parsed_output
                action["arguments"] = json.loads(action["arguments"])
                return action
            # invalid
            else:
                return None
        
        if "tool_calls" in message and message["tool_calls"] is not None and len(message["tool_calls"]) > 0 and message["tool_calls"][0]["function"] is not None:
            tool_call = message["tool_calls"][0]
            try:
                return {
                    "name": tool_call["function"]["name"].strip(),
                    "arguments": json.loads(tool_call["function"]["arguments"].strip()),
                }
            except json.JSONDecodeError:
                pass
        
        return None


    def base_action_parser(self, model_response):
        try:
            model_response = model_response.strip()
            origin_response = model_response
            action, action_input = None, None
            if 'Action:' in model_response:
                # try based on regex for action, action input
                action_pattern = r"Action:\s*(\w+)"
                action_match = re.search(action_pattern, model_response)
                action = action_match.group(1) if action_match else None
                if "Action Input:" in model_response:
                    action_input_match = model_response.split("Action Input:")[1].strip()
                    # if ```json``` is present, extract json content
                    pattern = r"```(?:json\s*)?\n*(\{.*?\})\n*```"
                    match = re.search(pattern, action_input_match, re.DOTALL)
                    if match:
                        action_input_match = match.group(1)
                    try:
                        action_input = json.loads(action_input_match) # load directly as json
                    except json.JSONDecodeError:
                        try:
                            action_input = ast.literal_eval(action_input_match)  # load as python object
                        except:
                            print('AST also failed')
                            pass
                if action_input is not None:
                    action_input = json.dumps(action_input) # cast to valid json string
                print("----", flush=True)
                print("\tAction:", action, flush=True)
                print("\tAction Input:", action_input, flush=True)
                print("----", flush=True)
            else:
                # assuming this is responding to the user
                return [model_response]
            assert action == None or isinstance(action, str)
            assert action_input == None or isinstance(action_input, str), f"it is {type(action_input)}"
            return [action, action_input]
        except:
            traceback.print_exc()
            return [None, None]
    
## custom llama parser
def parse_tool_response(response: str):
    function_regex = r"<function=(\w+)>(.*?)</function>"
    match = re.search(function_regex, response)
    if match:
        function_name, args_string = match.groups()
        try:
            args = json.loads(args_string)
            return {
                "function": function_name,
                "arguments": args,
            }
        except json.JSONDecodeError as error:
            print(f"Error parsing function arguments: {error}")
            return None
    return None
