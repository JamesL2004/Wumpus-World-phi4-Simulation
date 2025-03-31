import mesa
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from mesa.experimental.cell_space import CellAgent, FixedAgent
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from mesa.datacollection import DataCollector
from mesa.visualization import SolaraViz, make_plot_component, make_space_component

class HeroAgent(CellAgent):
    def __init__(self, model, cell):

        super().__init__(model)
        self.cell = cell

class WumpusAgent(FixedAgent):
    def __init__(self, model, cell):
        super().__init__(model)
        self.cell = cell

class PitAgent(FixedAgent):
    def __init__(self, model, cell):
        super().__init__(model)
        self.cell = cell

class GoldAgent(FixedAgent):
    def __init__(self, model, cell):
        super().__init__(model)
        self.cell = cell