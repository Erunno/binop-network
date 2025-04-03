import json
from network import Network
from exploration import NetworkExplorer, NetworkRandomEditor
from evaluation import NetworkEvaluator

class StochasticExplorationConfig:
    def __init__(self):
        self.samples_per_distance: list[tuple[int, int]] = []

    def set_samples_per_distance(self, distance: int, samples: int):
        self.samples_per_distance.append((distance, samples))
        return self
    
    def get_samples_per_distance(self) -> list[tuple[int, int]]:
        return sorted(self.samples_per_distance, key=lambda x: x[0])
    

class GradientDescent:
    def configure(self,
                  network: Network,
                  evaluator: NetworkEvaluator,
                  mix_up_coefficient: float = 0.1,
                  stochastic_exploration_config: StochasticExplorationConfig = None):
        
        self.network: Network = network
        self.evaluator: NetworkEvaluator = evaluator.set_network(network)
        self.mix_up_coefficient = mix_up_coefficient
        self.stochastic_exploration_config: StochasticExplorationConfig = stochastic_exploration_config

        return self

    def run(self, max_steps: int = 1000000):
        best_score = 0
        steps = 0
        did_step = True
        
        while best_score < 1 and steps < max_steps:
            stats = self.evaluator.get_statistics().replace('\n', ', ')
            print(f"Step {steps} -- stats - {stats}")
            
            if not did_step:
                # print(" -- Refreshing network, network score was:", self.evaluator.evaluate())
                # self._refresh_network()
                print(" -- Mixing up the network, network stats -", self.evaluator.get_statistics().replace('\n', ', '))
                self._mix_up_the_network()

            did_step = self._do_step()
            score = self.evaluator.evaluate()

            if score > best_score:
                best_score = score

                stats = self.evaluator.get_statistics().replace('\n', ', ')
                print (f"New best score - {best_score} (stats - {stats}), network:")
                print(self.network)
                print("JSON:")
                print(json.dumps(self.network.json()))
                print()

            steps += 1

        if steps >= max_steps:
            print("Max steps reached")

        print(f"Final score: {best_score}")
        print("Final network:")
        print(self.network)

    def _do_step(self):
        one_step_successful = self._do_one_change_step()
        if one_step_successful:
            return True
        
        print(" -- One step was not successful, trying stochastic exploration")

        stochastic_step_successful = self._do_stochastic_exploration_steps()

        if stochastic_step_successful:
            print(" -- Stochastic step was successful")

        return stochastic_step_successful

    def _do_one_change_step(self):
        explorer = NetworkExplorer().set_network(self.network)
        best_state = None
        best_score = self.evaluator.evaluate()

        while True:
            explorer.move_next()
            
            if explorer.explored_all:
                break

            score = self.evaluator.evaluate()

            if score > best_score:
                best_score = score
                best_state = explorer.export_state()

        if best_state is not None:
            explorer.adjust_network(best_state)
            return True

        return False
    
    def _do_stochastic_exploration_steps(self):
        if self.stochastic_exploration_config is None:
            return False
        
        random_editor = NetworkRandomEditor().set_network(self.network)
        best_score = self.evaluator.evaluate()
        best_changed_score = 0
        best_changes = None

        for distance, samples in self.stochastic_exploration_config.get_samples_per_distance():
            for _ in range(samples):
                random_editor.do_change_to_the_network(distance)

                score = self.evaluator.evaluate()
                if score > best_score:
                    best_score = score
                    best_changes = random_editor.get_changes()
                    print (" -- found better model:", self.evaluator.get_statistics().replace('\n', ', '))

                if score > best_changed_score:
                    best_changed_score = score

                random_editor.undo_changes()

        if best_changes is not None:
            best_changes.apply(self.network)
            return True
        else:
            print(" -- No better model found, best score was:", best_changed_score)
        
        return False

    def _refresh_network(self):
        self.network.refresh()

    def _mix_up_the_network(self):
        self.network.mix_up(change_coefficient=self.mix_up_coefficient)
