from __future__ import annotations

import os
import sys

from typing import TYPE_CHECKING

from utils.call_agent import ask_agent
from utils.ExperimentEnvironment import ExperimentEnvironment

class InterpAgent:
    """Agent that can execute code defined in an api txt file."""

    def __init__(
                self, 
                model_name: str, 
                prompt_path: str = "prompts",
                api_prompt_name: str = "api.txt",
                user_prompt_name: str = "user_mult.txt",
                overload_prompt_name: str = "final.txt",
                end_experiment_token: str = "[Difference]",
                max_round_count: int = 25,
                debug=False):
        self.model_name = model_name
        self.debug = debug
        self.experiment_log = []
        self.prompt_path = prompt_path
        self.api_prompt_name = api_prompt_name
        self.user_prompt_name = user_prompt_name
        self.overload_prompt_name = overload_prompt_name
        self.end_experiment_token = end_experiment_token
        self.max_round_count = max_round_count
    
    def _init_experiment_log(self):
        self.experiment_log = []
        with open(f'{self.prompt_path}/{self.api_prompt_name}', 'r') as file:
            api_prompt = file.read()
            self.update_experiment_log(role='system', type="text", type_content=api_prompt)
        with open(f'{self.prompt_path}/{self.user_prompt_name}', 'r') as file:
            user_prompt = file.read()
            self.update_experiment_log(role='user', type="text", type_content=user_prompt)

    def run_experiment(self, system: System, tools: Tools, save_html=False):
        """Runs the experiment loop. """

        # Make sure experiment log is clean
        self._init_experiment_log()
        experiment_env = ExperimentEnvironment(system, tools, globals())
        # Set Tools to point to this CodeAgent
        temp_agent, tools.agent = tools.agent, self
        # Experiment loop
        round_count = 0
        while True:
            round_count += 1
            model_experiment = ask_agent(self.model_name, self.experiment_log)
            self.update_experiment_log(role='model', type="text", type_content=str(model_experiment))
            if save_html:
                tools.generate_html(self.experiment_log)
            if self.debug:
                print(model_experiment)
            
            if round_count > self.max_round_count:
                self._overload_instructions()
            else:
                if self.end_experiment_token in model_experiment:
                    break
                
                try:
                    experiment_output = experiment_env.execute_experiment(model_experiment)
                    if experiment_output != "":
                        self.update_experiment_log(role='user', type="text", type_content=experiment_output)
                except ValueError:
                    self.update_experiment_log(role='execution', 
                                               type="text", 
                                               type_content=f"No code to run was provided, please continue with the experiments based on your findings, or output your final {self.end_experiment_token}.")
        if save_html:
            tools.generate_html(self.experiment_log)
        
        # Restore tools to its original state
        tools.agent = temp_agent

    def update_experiment_log(self, role, content=None, type=None, type_content=None):
        openai_role = {'execution':'user','model':'assistant','user':'user','system':'system'}
        if type == None:
            self.experiment_log.append({'role': openai_role[role], 'content': content})
        elif content == None:
            if type == 'text':
                self.experiment_log.append({'role': openai_role[role], 'content': [{"type":type, "text": type_content}]})
            if type == 'image_url':
                self.experiment_log.append({'role': openai_role[role], 'content': [{"type":type, "image_url": type_content}]})

    def _overload_instructions(self):
        with open(f'{os.path.join(self.prompt_path, self.overload_prompt_name)}', 'r') as file:
            final_instructions = file.read()
            self.update_experiment_log(role='user', type="text", type_content=final_instructions)

if TYPE_CHECKING:
    from maia_api import System, Tools