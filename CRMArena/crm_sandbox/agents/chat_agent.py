import os
from litellm import completion
import litellm
litellm.set_verbose = False
from typing import Dict, List
import time, traceback
from crm_sandbox.agents.prompts import SCHEMA_STRING, REACT_RULE_STRING, SYSTEM_METADATA, REACT_EXTERNAL_INTERACTIVE_PROMPT, REACT_INTERNAL_INTERACTIVE_PROMPT, REACT_INTERNAL_PROMPT, REACT_EXTERNAL_PROMPT, REACT_PRIVACY_AWARE_EXTERNAL_PROMPT, REACT_PRIVACY_AWARE_EXTERNAL_INTERACTIVE_PROMPT, ACT_PROMPT
from crm_sandbox.agents.utils import parse_wrapped_response, BEDROCK_MODELS_MAP, TOGETHER_MODELS_MAP, VERTEX_MODELS_MAP
import together





class ChatAgent:
    def __init__(
        self, schema_obj, model: str = "gpt-4o", max_turns: int = 20, eval_mode="default", strategy="react", provider="bedrock", interactive=False, agent_type="internal", privacy_aware_prompt=False
    ):
        schema = self._build_schema(schema_obj)
        assert strategy in ["react", "act"], "Only react and act strategies supported for now"
        assert agent_type in ["internal", "external"], "Invalid agent type"
        
        if strategy == "react":
            # react strategy
            if not interactive:
                if agent_type == "internal":
                    self.sys_prompt = REACT_INTERNAL_PROMPT.format(system_description=schema, system="Salesforce instance") # add strategy template and schema description
                else:
                    if privacy_aware_prompt:
                        self.sys_prompt = REACT_PRIVACY_AWARE_EXTERNAL_PROMPT.format(system_description=schema, system="Salesforce instance") # add strategy template and schema description
                    else:
                        self.sys_prompt = REACT_EXTERNAL_PROMPT.format(system_description=schema, system="Salesforce instance") # add strategy template and schema description
            else:
                if agent_type == "internal":
                    self.sys_prompt = REACT_INTERNAL_INTERACTIVE_PROMPT.format(system_description=schema, system="Salesforce instance") # add strategy template and schema description
                else:
                    if privacy_aware_prompt:
                        self.sys_prompt = REACT_PRIVACY_AWARE_EXTERNAL_INTERACTIVE_PROMPT.format(system_description=schema, system="Salesforce instance") # add strategy template and schema description
                    else:
                        self.sys_prompt = REACT_EXTERNAL_INTERACTIVE_PROMPT.format(system_description=schema, system="Salesforce instance") # add strategy template and schema description
        else:
            # act strategy
            self.sys_prompt = ACT_PROMPT.format(system_description=schema, system="Salesforce instance")
        
        self.agent_type = agent_type
        self.original_model_name = model
        self.model = model
        self.eval_mode = eval_mode
        self.max_turns = max_turns
        self.strategy = strategy
        self.info = {}
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
            pass
        if self.model in ["o1-mini", "o1-preview", "o1-2024-12-17", "o3-mini-2025-01-31"]:
            import litellm
            
            litellm.drop_params=True
            print("dropping parameters")
            
            # assert self.model in ["o1-mini", "o1-preview", "gpt-4o-2024-08-06", "gpt-3.5-turbo-0125"], "Invalid model name"
    
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
        if self.original_model_name not in ["o1-mini", "o1-preview", "o1-2024-12-17", "deepseek-r1", "o3-mini-2025-01-31", "gemini-2.5-flash-preview-04-17"]:
            self.messages = [{"role": "system", "content": self.sys_prompt.strip()}]
            self.messages.append({"role": "user", "content": args["query"].strip()})
        
        else:
            # No system role for o1-mini and o1-preview
            self.messages = [{"role": "user", "content": self.sys_prompt + "\n\n" + args["query"]}]
        self.usage = {"cost": [], "completion_tokens": [], "prompt_tokens": [], "total_tokens": []}
        
    def act(self, env, index=None, temperature=0.0):
        query, metadata = env.reset(task_index=index)
        self.reset({"query": query, "metadata": metadata})
        # print("----")
        # print(self.sys_prompt)
        # print("----")
        total_cost = 0.0
        self.info["observation_sizes"] = []
        done = False
        reward = 0
        
        current_agent_turn = 0
        # for turn_id in range(self.max_turns):
        while current_agent_turn < self.max_turns:
            # sleep for non-openai models
            if self.provider != "openai":
                time.sleep(5)
            info = {}
            current_agent_turn += 1
            # turn off thinking for gemini 2.5 flash
            if self.original_model_name == "gemini-2.5-flash-preview-04-17":
                thinking = {"type": "disabled", "budget_tokens": 0}
            elif self.original_model_name == "gemini-2.5-flash-preview-04-17-thinking-4096":
                thinking = {"type": "enabled", "budget_tokens": 4096}
            else:
                thinking = None
            
            res = completion(
                messages=self.messages,
                model=self.model,
                temperature=0.0,
                max_tokens=2000 if self.original_model_name not in ["o1-mini", "o1-preview", "o1-2024-12-17", "deepseek-r1", "o3-mini-2025-01-31", "gemini-2.5-flash-preview-04-17", "gemini-2.5-flash-preview-04-17-thinking-4096", "gemini-2.5-pro-preview-03-25"] else 50000,
                top_p=1.0 if self.model not in ["o3-mini-2025-01-31"] else None,
                thinking= thinking,  
                # custom_llm_provider=self.provider,
                additional_drop_params=["temperature"] if self.original_model_name in ["o1-mini", "o1-preview", "o1-2024-12-17", "deepseek-r1", "o3-mini-2025-01-31"] else []
            )

            
            
            message = res.choices[0].message.model_dump()
            
            
            usage = res.usage

            for key in self.usage.keys():
                if key != "cost":
                    self.usage[key].append(usage.get(key, 0))

            self.usage["cost"].append(res._hidden_params["response_cost"])
            action = self.message_action_parser(message, self.model)
            print("User Turn:", env.current_user_turn, "Agent Turn:", current_agent_turn, "Agent:", message["content"].strip())
            self.messages.append({"role": "assistant", "content": message["content"].strip()})
            if action is None:
                self.info["end_reason"] = {
                    "source": "agent",
                    "message": "Invalid action",
                    "content":  message["content"].strip()
                }
                info["end_reason"] = self.info["end_reason"]
                if self.strategy == "react":
                    self.messages.append({"role": "user", "content": REACT_RULE_STRING})
                elif self.strategy == "act":
                    self.messages.append({"role": "user", "content": ACT_RULE_STRING})
                continue
            obs, reward, done, info = env.step(action)
            
            if "observation_size" in info:
                self.info["observation_sizes"].append(info["observation_size"])
            if "end_reason" in info: # implies error in query
                self.info["end_reason"] = info["end_reason"]
            # reset counter if previous action is respond
            if action["name"] == "respond":
                current_agent_turn = 0
            if done:
                break
            elif action["name"] == "execute": # execution results from
                self.messages.append({"role": "user", "content": f"Salesforce instance output: {obs}"})
            elif action["name"] == "respond": # respond to simulated user
                self.messages.append({"role": "user", "content": obs})
        
        # Here when either max_turns is reached or submitted
        if not done: 
            if "end_reason" not in info: # no error in last query
                self.info["end_reason"] = {
                    "source": "agent",
                    "message": "Max turns reached",
                    "content":  message["content"].strip()
                }
        self.info["usage"] = self.usage
        self.info["total_cost"] = sum(cost for cost in self.usage["cost"] if cost is not None)
        self.info["num_turns"] = (env.current_user_turn, current_agent_turn + 1)
        return reward

    def get_messages(self) -> List[Dict[str, str]]:
        return self.messages

    @staticmethod
    def message_action_parser(message: str, model_name: str) -> Dict[str, str]:
        action = None
        content = message["content"].strip()
        # if model_name "deepseek-r1":
        #     content = content.split("</think>")[1]
        resp = parse_wrapped_response(r'<execute>(.*?)</execute>', content).strip()
        if resp:
            action = {"name": "execute", "content": resp}
            return action
        
        resp = parse_wrapped_response(r'<respond>(.*?)</respond>', content).strip()
        if resp:
            action = {"name": "respond", "content": resp}
            return action
        return action