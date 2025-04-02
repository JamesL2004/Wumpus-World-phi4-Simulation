import mesa
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from mesa.experimental.cell_space import CellAgent, FixedAgent
from mesa.experimental.devs import ABMSimulator
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from mesa.visualization import SolaraViz, make_plot_component, make_space_component

class HeroAgent(mesa.Agent):
    def __init__(self, model):

        super().__init__(model)
    def step(self):
        self.move()
        message = self.get_Effects() 

        output = self.model.PromptModel(
            "You are a hero in a simulation that is looking to find the gold to finish the game, but there are obstacles in your way. "
            "There are pits and the wumpus; if you encounter either, you fail and die. But you can tell if they are nearby by the effects they leave on neighboring tiles. "
            "Pits create a breeze, and the Wumpus creates a foul smell. The gold also leaves a glitter effect nearby, so if you encounter these effects, think carefully about your next step.",
            message,
            "Provide a single sentance about your current position: "
        )
        print(output)

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            tuple(self.pos), 
            moore=False,
            include_center=False
        )
        print(possible_steps)
        print(self.get_Directions(possible_steps))
        
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def get_Effects(self):
        message = ""
        effects = self.model.effects[tuple(self.pos)]
        print(effects)
        if effects["smell"]:
            message += "You smell a foul smell coming from a neighbouring tile. "
        if effects["breeze"]:
            message += "You feel a breeze coming from a neighbouring tile. "
        if effects["smell"] is False and effects["breeze"] is False:
            message = "There are no effects on this cell"
        return message
    
    def get_Directions(self, possible_steps):

        possible_directions = ["left", "right", "up", "down"]

        for step in possible_steps:
            if step[0] == 0:
                possible_directions.remove("left")
            if step[1] == 0:
                possible_directions.remove("down")
            if step[0] == (self.model.width - 1):
                possible_directions.remove("right")
            if step[1] == (self.model.height - 1):
                possible_directions.remove("up")

        return possible_directions

class WumpusAgent(FixedAgent):
    def __init__(self, model):
        super().__init__(model)

class PitAgent(FixedAgent):
    def __init__(self, model):
        super().__init__(model)

class GoldAgent(FixedAgent):
    def __init__(self, model):
        super().__init__(model)

class WumpusModel(mesa.Model):

    def __init__(
        self,
        width=4,
        height=4,
        pits=3,
        gold=1,
        wumpus=1,   
    ):
        
        super().__init__()
        self.width = width
        self.height = height
        self.grid = mesa.space.MultiGrid(width, height, False)
        self.effects = {(x, y): {"smell": False, "breeze": False, "glitter": False} for x in range(width) for y in range(height)}

        heroagent = HeroAgent(self)
        self.grid.place_agent(heroagent, [0, 0])
        empty_cells = [(x, y) for x in range(width) for y in range(height)]

        empty_cells.remove((0, 0))

        def place_agents(agent_class, population):
            nonlocal empty_cells
            position = self.random.sample(empty_cells, population)
            for pos in position:
                agent = agent_class(self)
                self.grid.place_agent(agent, tuple(pos))
                self.update_effects(tuple(pos), agent)
                empty_cells.remove(pos)
        place_agents(PitAgent, pits)
        place_agents(WumpusAgent, wumpus)
        place_agents(GoldAgent, gold)

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(i, torch.cuda.get_device_properties(i))

        torch.random.manual_seed(0)

        model_path = "microsoft/Phi-4-mini-instruct"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cpu", # "cpu" or "auto" or "cuda:0" for cuda device 0, 1, 2, 3 etc. if you have multiple GPUs
            torch_dtype="auto",
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        #gotta have a tokenizer for each model otherwise the token mappings won't match
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )

    def update_effects(self, pos, agentType):
        neighbors = self.grid.get_neighborhood(pos, moore=False, include_center=False)
        for n in neighbors:
            if n in self.effects:  # Ensure it's a valid grid position
                if isinstance(agentType, PitAgent):
                    self.effects[n]["breeze"] = True
                elif isinstance(agentType, WumpusAgent):
                    self.effects[n]["smell"] = True
    def PromptModel(self, context, memorystream, prompt):
        
        #lower temperature generally more predictable results, you can experiment with this
        generation_args = {
            "max_new_tokens": 64,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }

        llmprompt = prompt
        
        messages = [
            {"role": "system", "content": context},
            {"role": "system", "content": memorystream},
            {"role": "user", "content": llmprompt},
        ]

        #time1 = int(round(time.time() * 1000))

        output = self.pipe(messages, **generation_args)
        return output[0]['generated_text']

        #time2 = int(round(time.time() * 1000))
        #print("Generation time: " + str(time2 - time1))
        #self.datacollector.collect(self)
    def step(self):
        self.agents_by_type[HeroAgent].shuffle_do("step")


def wumpus_portrayal(agent):
    if agent is None:
        return
    
    portrayal = {
        "size": 25
    }

    if isinstance(agent, WumpusAgent):
        portrayal["color"] = "tab:green"
        portrayal["marker"] = "o"
        portrayal["zorder"] = 2
    elif isinstance(agent, PitAgent):
        portrayal["color"] = "tab:cyan"
        portrayal["marker"] = "o"
        portrayal["zorder"] = 2
    elif isinstance(agent, GoldAgent):
        portrayal["color"] = "tab:blue"
        portrayal["marker"] = "o"
        portrayal["zorder"] = 2
    elif isinstance(agent, HeroAgent):
        portrayal["color"] = "tab:red"
        portrayal["marker"] = "o"
        portrayal["zorder"] = 2
    
    return portrayal

model_params = {
    "width": {
        "type": "SliderInt",
        "value": 20,
        "label": "Width:",
        "min": 10,
        "max": 100,
        "step": 10,
    },
    "height": {
        "type": "SliderInt",
        "value": 20,
        "label": "Height:",
        "min": 10,
        "max": 100,
        "step": 10,
    },
}

wumpus_model = WumpusModel(4, 4, 3, 1, 1)
SpaceGraph = make_space_component(wumpus_portrayal)

page = SolaraViz(
    wumpus_model,
    components=[SpaceGraph],
    model_params=model_params,
    name="Wumpus World"
)

page