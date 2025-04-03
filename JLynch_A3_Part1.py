import mesa
import sys
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
    def __init__(self, model, move_history, arrow):

        super().__init__(model)
        self.move_history = move_history
        self.arrow = arrow
    def step(self):
        message = self.get_Effects() 

        output = self.model.PromptModel(
            "You are a hero in a simulation that is looking to find the gold to finish the game, but there are obstacles in your way. "
            "There are pits and the wumpus; if you encounter either, you fail and die. But you can tell if they are nearby by the effects they leave on neighboring tiles. "
            "Pits create a breeze, and the Wumpus creates a foul smell. The gold also leaves a glitter effect nearby, so if you encounter these effects, think carefully about your next step.",
            message,
            "Provide a single sentance about your current position: "
        )
        current_effects = self.model.effects[tuple(self.pos)]
        if current_effects["smell"] and self.arrow == 1:
            output = self.model.PromptModel(
                "You are a hero in a simulation that is looking to find the gold to finish the game, but there are obstacles in your way. "
                "There are pits and the wumpus; if you encounter either, you fail and die. But you can tell if they are nearby by the effects they leave on neighboring tiles. "
                "Pits create a breeze, and the Wumpus creates a foul smell. The gold also leaves a glitter effect nearby, so if you encounter these effects, think carefully about your next step.",
                "There is a foul smell nearby which means there is a wumpus in a adjancent tile to you if you want to you can shoot at one of those tiles for a chance to kill the Wumpus. But you only have one shot available so use it wisely.",
                "Would you like to try and shoot the wumpus. Reply by only responding with yes or no: "
            )
            print(output)
            if output.strip().lower() == "yes":
                possible_directions = self.get_Directions(self.pos)
                output = self.model.PromptModel(
                    "You are a hero in a simulation that is looking to find the gold to finish the game, but there are obstacles in your way. "
                    "There are pits and the wumpus; if you encounter either, you fail and die. But you can tell if they are nearby by the effects they leave on neighboring tiles. "
                    "Pits create a breeze, and the Wumpus creates a foul smell. The gold also leaves a glitter effect nearby, so if you encounter these effects, think carefully about your next step.",
                    "You chose to try and shoot the wumpus the following are the possible directions you can shoot " + str(possible_directions),
                    "Based on the possible directions pick a single direction, only output the word of a single direction no puncuation:"
                )
                possible_steps = self.model.grid.get_neighborhood(
                    tuple(self.pos), 
                    moore=False,
                    include_center=False
                )
                next_step = self.get_Next_Step(output, possible_directions, possible_steps)
                shot_cell_contents = self.model.grid.get_cell_list_contents(next_step[0])
                for agent in shot_cell_contents:
                    if isinstance(agent, WumpusAgent):
                        agent.dead = True
                        print("You just shot the Wumpus.")
                        self.arrow = 0
                    else:
                        print("You failed at shooting the Wumpus")
        print(output)
        self.move()
        self.check_Cell()
    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            tuple(self.pos), 
            moore=False,
            include_center=False
        )
        #print(all_directions)
        possible_directions = self.get_Directions(self.pos)
        print(possible_directions)
        #print(self.pos)
        message = self.get_Effects()
        last_move = self.move_history[-1]
        output = self.model.PromptModel(
            "You are a hero in a simulation that is looking to find the gold to finish the game your current position is " + str(self.pos) + "."
            "There are pits and the wumpus; if you encounter either, you fail and die. But you can tell if they are nearby by the effects they leave on neighboring tiles. "
            "Pits create a breeze, and the Wumpus creates a foul smell. So if you encounter these effects, think carefully about your next step. "
            "The opposite of left is right and the opposite of up is down, so if you  sense a breeze or smell you can backtrack and try a different direction. "
            "You can do the same for all the directions if they are available"
            "On the map you can usually move left, right, up or down, but these can chagne depending on where you are. So make sure you try all the directions to cover the whole map. "
            "A good strategy is if you don't sense a breeze or a smell go in that direction again, but if you do sense one of them than backtrack and take a different route.",
            message + " These are your past moves " + str(self.move_history) + " with " + str(last_move) + " being your last move, try not to repeat the same sequence of moves so you don't end up in a loop",
            "The following are the possible directions you can move " + (str(possible_directions)) + " based on the effects you can feel and your previous moves choose one of the directions, "
            "only output the word of the single direction no punctuation: "
        )
        print(output)
        self.move_history.append(output)
        print(self.move_history)
        next_step = self.get_Next_Step(output, possible_directions, possible_steps)
        new_position = self.random.choice(next_step)
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
    
    def get_Directions(self, position):

        possible_directions = ["left", "down", "up", "right"]

        if position[0] == 0 and "left" in possible_directions:
            possible_directions.remove("left")
        elif position[0] == (self.model.width - 1) and "right" in possible_directions:
            possible_directions.remove("right")
        if position[1] == 0 and "down" in possible_directions:
            possible_directions.remove("down")
        elif position[1] == (self.model.height - 1) and "up" in possible_directions:
            possible_directions.remove("up")

        return possible_directions
    
    def get_Next_Step(self, choice, directions, possible_steps):
        index = directions.index(choice)
        return [possible_steps[index]]
    
    def check_Cell(self):
        other_agent = self.model.grid.get_cell_list_contents([self.pos])
        
        for agent in other_agent:
        
            if isinstance(agent, PitAgent):
                print("You failed by falling into a pit.")   
                sys.exit()          
            elif isinstance(agent, WumpusAgent) and agent.dead is False:
                print("You failed by running into the Wumpus.") 
                sys.exit()
            elif isinstance(agent, GoldAgent):
                print("YOU DID IT, you found the gold! Congratulations!")
                sys.exit()

class WumpusAgent(mesa.Agent):
    def __init__(self, model):
        super().__init__(model)

        self.dead = False

class PitAgent(mesa.Agent):
    def __init__(self, model):
        super().__init__(model)

class GoldAgent(mesa.Agent):
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
        self.pits = pits
        self.wumpus = wumpus
        self.gold = gold
        self.grid = mesa.space.MultiGrid(width, height, False)
        self.effects = {(x, y): {"smell": False, "breeze": False, "glitter": False} for x in range(width) for y in range(height)}

        heroagent = HeroAgent(self, ["none"], 1)
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
            "temperature": 0.5,
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