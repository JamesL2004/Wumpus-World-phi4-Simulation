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

class HeroAgent(CellAgent):
    def __init__(self, model):

        super().__init__(model)

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
        hero=1,    
    ):
        
        super().__init__()
        self.grid = mesa.space.MultiGrid(width, height, True)

        heroagent = HeroAgent(self)
        self.grid.place_agent(heroagent, [0, 0])
        empty_cells = [(x, y) for x in range(width) for y in range(height)]

        empty_cells.remove((0, 0))

        def place_agents(agent_class, population):
            nonlocal empty_cells
            position = self.random.sample(empty_cells, population)
            for pos in position:
                agent = agent_class(self)
                self.grid.place_agent(agent, pos)
                empty_cells.remove(pos)
                print(pos)
        place_agents(PitAgent, pits)
        place_agents(WumpusAgent, wumpus)
        place_agents(GoldAgent, gold)


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

wumpus_model = WumpusModel(4, 4, 3, 1, 1, 1)
SpaceGraph = make_space_component(wumpus_portrayal)

page = SolaraViz(
    wumpus_model,
    components=[SpaceGraph],
    model_params=model_params,
    name="Wumpus World"
)

page