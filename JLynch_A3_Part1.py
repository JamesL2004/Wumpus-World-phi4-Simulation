import mesa
import sys
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from mesa.visualization import SolaraViz, make_plot_component, make_space_component

class HeroAgent(mesa.Agent):
    def __init__(self, model):
        super().__init__(model)
        self.move_history = {}
        self.arrow = 1
        self.step_count = 0

    def step(self):
        self.step_count += 1
        message = self.get_Effects()
        last_move = list(self.move_history.keys())[-1] if self.move_history else "none"
        print(last_move)

        internal_dialogue = self.model.PromptModel(
            "You are a hero in a simulation that is looking to find the gold to finish the game, but there are obstacles in your way. "
            f"Your current position is {self.pos}. "
            "There are pits and the wumpus; if you encounter either, you fail and die. But you can tell if they are nearby by the effects they leave on neighboring tiles. "
            "Pits create a breeze, and the Wumpus creates a foul smell. The gold also leaves a glitter effect nearby, so if you encounter these effects, think carefully about your next step.",
            f"{message} Here is your move history: {self.move_history}. Your last move was {last_move}, so try not to repeat it unless necessary.",
            "Provide a detailed explanation of your journey so far and your current position with a 30 word limit: "
        )

        current_effects = self.model.effects[tuple(self.pos)]
        if current_effects["smell"] and self.arrow == 1:
            shoot_prompt = self.model.PromptModel(
                "You are a hero in a simulation that is looking to find the gold to finish the game, but there are obstacles in your way. "
                "There are pits and the wumpus; if you encounter either, you fail and die. But you can tell if they are nearby by the effects they leave on neighboring tiles. "
                "Pits create a breeze, and the Wumpus creates a foul smell. The gold also leaves a glitter effect nearby, so if you encounter these effects, think carefully about your next step.",
                "There is a foul smell nearby which means there is a wumpus in a adjancent tile to you if you want to you can shoot at one of those tiles for a chance to kill the Wumpus. But you only have one shot available so use it wisely.",
                "Would you like to try and shoot the wumpus. Reply by only responding with yes or no: "
            )
            if shoot_prompt.strip().lower() == "yes":
                possible_directions = self.get_Directions(self.pos)
                shoot_dir = self.model.PromptModel(
                    "You are a hero in a simulation that is looking to find the gold to finish the game, but there are obstacles in your way. "
                    "There are pits and the wumpus; if you encounter either, you fail and die. But you can tell if they are nearby by the effects they leave on neighboring tiles. "
                    "Pits create a breeze, and the Wumpus creates a foul smell. The gold also leaves a glitter effect nearby, so if you encounter these effects, think carefully about your next step.",
                    f"You chose to try and shoot the wumpus the following are the possible directions you can shoot {possible_directions}",
                    "Based on the possible directions pick a single direction, only output the word of a single direction no punctuation:"
                )
                possible_steps = self.model.grid.get_neighborhood(
                    tuple(self.pos), 
                    moore=False,
                    include_center=False
                )
                next_step = self.get_Next_Step(shoot_dir, possible_directions, possible_steps)
                shot_cell_contents = self.model.grid.get_cell_list_contents(next_step[0])
                for agent in shot_cell_contents:
                    if isinstance(agent, WumpusAgent):
                        agent.dead = True
                        self.model.update_effects(agent.pos, agent, False)
                        internal_dialogue += "\nYou successfully shot the Wumpus!"
                        break
                else:
                    internal_dialogue += "\nYou failed to hit the Wumpus."
                self.arrow = 0

        previous_position = self.pos
        with open(f"hero_journey.txt", "a") as f:
            f.write(f"Step #{self.step_count}\n")
            f.write(f"Previous Position: {previous_position}\n")
        self.move()
        current_position = self.pos
        message = self.get_Effects()

        with open(f"hero_journey.txt", "a") as f:
            f.write(f"New Position: {current_position}\n")
            f.write(f"Effects sensed: {message}\n")
            f.write(f"Internal Dialogue: {internal_dialogue}\n")
        
        self.check_Cell()

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            tuple(self.pos), 
            moore=False,
            include_center=False
        )
        possible_directions = self.get_Directions(self.pos)
        
        message = self.get_Effects()
        last_move = list(self.move_history.keys())[-1] if self.move_history else "none"

        print(f"Possible directions: {possible_directions}")
        print(f"Last move: {last_move}")

        output = self.model.PromptModel(
            "You are a hero in a simulation that is looking to find the gold to finish the game. "
            f"Your current position is {self.pos}. "
            "There are pits and the Wumpus; if you encounter either, you fail and die. "
            "Pits create a breeze, and the Wumpus creates a foul smell. So if you encounter these effects, think carefully about your next step. "
            "A good strategy is if you don't sense a breeze or a smell, go in that direction again. "
            "Otherwise, backtrack and take a different route.",
            f"{message} Here is your move history: {self.move_history}. Your last move was {last_move}. Try not to repeat the same sequence of directions so your not going in a loop",
            f"The possible directions you can move are {possible_directions}. Based on the effects you sense and your previous moves, choose a direction. "
            "Only output the word of the direction with no punctuation: "
        )

        print(output)

        with open(f"hero_journey.txt", "a") as f:
            f.write(f"Moved: {output}\n")

        if output not in possible_directions:
            output = self.random.choice(possible_directions)

        next_step = self.get_Next_Step(output, possible_directions, possible_steps)
        new_position = self.random.choice(next_step)

        self.move_history[output] = new_position 
        self.model.grid.move_agent(self, new_position)

    def get_Effects(self):
        message = ""
        effects = self.model.effects[tuple(self.pos)]
        if effects["smell"]:
            message += "You smell a foul smell coming from a neighboring tile. "
        if effects["breeze"]:
            message += "You feel a breeze coming from a neighboring tile. "
        if not effects["smell"] and not effects["breeze"]:
            message = "There are no effects on this cell."
        return message

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

    def check_Cell(self):
        other_agent = self.model.grid.get_cell_list_contents([self.pos])
        
        for agent in other_agent:
            if isinstance(agent, PitAgent):
                with open(f"hero_journey.txt", "a") as f:
                    f.write("You failed by falling into a pit.")  
                sys.exit()          
            elif isinstance(agent, WumpusAgent) and agent.dead is False:
                with open(f"hero_journey.txt", "a") as f:
                    f.write("You failed by hitting the Wumpus.") 
                sys.exit()
            elif isinstance(agent, GoldAgent):
                with open(f"hero_journey.txt", "a") as f:
                    f.write("YOU DID IT! You found the goal congrats.")
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
                self.update_effects(tuple(pos), agent, True)
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

    def update_effects(self, pos, agentType, active):
        neighbors = self.grid.get_neighborhood(pos, moore=False, include_center=False)
        for n in neighbors:
            if n in self.effects:  # Ensure it's a valid grid position
                if isinstance(agentType, PitAgent):
                    self.effects[n]["breeze"] = active
                elif isinstance(agentType, WumpusAgent):
                    self.effects[n]["smell"] = active
        if isinstance(agentType, GoldAgent):
            self.effects[pos]["glitter"] = active
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