import mesa
import seaborn as sns
import numpy as np
import pandas as pd
import random as rn
import torch
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from mesa.datacollection import DataCollector
from mesa.visualization import SolaraViz, make_plot_component, make_space_component

def compute_gini(model):
    return sum(1 for agent in model.agents if not agent.isZombie and not agent.dead) 

class OutbreakAgent(mesa.Agent):

    def __init__(self, model):
        super().__init__(model)

        self.isZombie = False
        self.shotsLeft = 15
        self.dead = False
        self.cureShots = 10
        self.hitPoints = 3
        self.past_events = {}
        self.past_conversations = {}
        self.step_count = 0

    def step(self):
        if self.dead == True:
            return
        
        self.step_count += 1
        all_agents = list(self.model.grid.agents)
        zombie_count = 0
        for agent in all_agents:
            if agent.isZombie == True:
                zombie_count += 1
        self.move()

        
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        humans = []
        for cell in cellmates:
            if cell != self and not cell.isZombie:
                humans.append(cell)
        if humans: 
            otherHuman = self.random.choice(humans)
            otherHuman.responder = True
            self.haveConversation(otherHuman)
    

        if self.isZombie == True:
            self.infect()
            if rn.random() < 0.5:
                self.dropAmmo()
        elif self.isZombie == False:
            self.shootZombie()

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=False,
            include_center=False)
        
        if self.isZombie == False:
            possible_directions = self.get_Directions(self.pos)
            all_agents = list(self.model.grid.agents)
            zombie_count = 0
            for agent in all_agents:
                if agent.isZombie == True:
                    zombie_count += 1
            output = self.model.PromptModel(
                f"Your name is Human {self.unique_id}, and you are currently in the middle of a zombie apocalypse where there is currently {zombie_count} zombies left."
                f"Your goal is to keep moving to avoid the zombies and try and talk to other humans to plan. You current position is {self.pos}",
                f"Based on your current position your available directions to move in are {possible_directions}, also the following list are important past events that happened and what tile they happened on {self.past_events}",
                "Based on the past events and some randomness to avoid repetition provide a direction to move in next. Provide only the single direction word by itself, no puncuation:"
            )
            print(output)
            next_step = self.get_Next_Step(output, possible_directions, possible_steps)
            new_position = self.random.choice(next_step)
            self.model.grid.move_agent(self, new_position)
        else:
            new_position = self.random.choice(possible_steps)
            self.model.grid.move_agent(self, new_position)
    
    def get_Directions(self, position):
        possible_directions = ["left", "down", "up", "right"]
        if position[0] == 0:
            possible_directions.remove("left")
        elif position[0] == (self.model.width - 1):
            possible_directions.remove("right")
        if position[1] == 0:
            possible_directions.remove("down")
        elif position[1] == (self.model.height - 1):
            possible_directions.remove("up")
        return possible_directions

    def get_Next_Step(self, choice, directions, possible_steps):
        index = directions.index(choice)
        return [possible_steps[index]]

    def infect(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        humans = []
        for cell in cellmates:
            if cell.isZombie == False:
                humans.append(cell)
        if len(humans) > 0:
             other = self.random.choice(humans)
             other.isZombie = True
    
    def dropAmmo(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        humans = []
        for cell in cellmates:
            if cell.isZombie == False:
                humans.append(cell)
        if len(humans) > 0:
            other = self.random.choice(humans)
            self.shotsLeft -= 3
            other.shotsLeft += 3

    def shootZombie(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        zombies = []
        for cell in cellmates:
            if cell.isZombie == True:
                zombies.append(cell)
        if rn.random() < 0.5:
            if self.shotsLeft > 0:
                if zombies: 
                    other = self.random.choice(zombies)
                    if other.hitPoints > 1:
                        other.hitPoints -= 1 
                        self.shotsLeft -= 1
                    else:
                        other.dead = True
                        self.past_events[self.pos] = f"You killed Zombie {other.unique_id} on this position"
        elif rn.random() < 0.7:
            if self.cureShots > 0:
                if zombies:
                    other = self.random.choice(zombies)
                    other.isZombie = False
                    self.cureShots -= 1
    
    def haveConversation(self, other):

        all_agents = list(self.model.grid.agents)
        zombie_count = 0
        for agent in all_agents:
            if agent.isZombie == True:
                zombie_count += 1

        human_conversation = self.model.PromptModel(
            f"Your name is Human {self.unique_id}, and you are currently in the middle of a zombie apocalypse where there is currently {zombie_count} zombies left."
            f"Your goal is to keep moving to avoid the zombies and try and talk to other humans to plan. You current position is {self.pos}, You have the ability to shoot zombies if you have enough ammo, you currently have {self.cureShots} bullets left.",
            f"You just ran into Human {other.unique_id}, This list is any important events that has happened to you {self.past_events} and this is the events that have happened to Human {other.unique_id}, {other.past_events}, "
            f"This list is all the past conversations you Human {self.unique_id} has had: {self.past_conversations}, and this is the past conversations Human {other.unique_id} has had: {other.past_conversations}"
            F"If you have had a conversation with Human {other.unique_id}, use the last conversation in {self.past_conversations} you've had with them for context.",
            f"Generate a conversation between you: Human {self.unique_id} and the other human {other.unique_id}, about some interactions both of you have had in the world, and how you are currently doing in the simulation: "
        )
        print(human_conversation)
        with open(f"a3_jlynch_part_c_chat.txt", "a") as f:
            f.write(f"Step#: {self.step_count}\n")
            f.write(f"Zombie's Left: {zombie_count}\n")
            f.write(f"Conversation taken place at {self.pos}\n")
            f.write(f"{human_conversation}\n")
        self.past_conversations[other.unique_id] = human_conversation
        other.past_conversations[self.unique_id] = human_conversation


class OutbreakModel(mesa.Model):
    """A model with some number of agents."""
    def __init__(self, totalAgents=100, width=8, height=8):
        super().__init__()
        self.total_agents = 10
        self.width = width
        self.height = height
        self.grid = mesa.space.MultiGrid(width, height, False)
        self.datacollector = mesa.DataCollector(
            model_reporters={"Humans Left": compute_gini}
        )
        # Create agents
        for i in range(self.total_agents):
            agent = OutbreakAgent(self)

            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

            if rn.random() < 0.3:  # 10% chance
                agent.isZombie = True

        self.running = True
        #self.datacollector.collect(self)
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

    def PromptModel(self, context, memorystream, prompt):
        
        #lower temperature generally more predictable results, you can experiment with this
        generation_args = {
            "max_new_tokens": 64,
            "return_full_text": False,
            "temperature": 1.0,
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

    def step(self):
        """Advance the model by one step."""
        self.datacollector.collect(self)
        self.agents.shuffle_do("step")


model_params = {
    "totalAgents": {
        "type": "SliderInt",
        "value": 50,
        "label": "Number of agents:",
        "min": 10,
        "max": 100,
        "step": 1,
    },
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

#modify this function to change output on grid
def agent_portrayal(agent):
    size = 20
    color = "tab:red"

    if agent.dead == True:
        size = 40
        color = "tab:blue"
    elif agent.isZombie == True:
        size = 60
        color = "tab:green"
    elif agent.isZombie == False:
        size = 20
        color ="tab:red"
    return {"size": size, "color": color}

outbreak_model = OutbreakModel(10, 8, 8)

SpaceGraph = make_space_component(agent_portrayal)
GiniPlot=make_plot_component("Humans Left")

page = SolaraViz(
    outbreak_model,
    components=[SpaceGraph, GiniPlot],
    model_params=model_params,
    name="Zombie Outbreak Model"
)
# This is required to render the visualization in the Jupyter notebook
page