# Set up the game environment
import rlcard
from rlcard.agents import NolimitholdemHumanAgent as HumanAgent
from rlcard.envs import Env
from rlcard.games.nolimitholdem import Game
from rlcard.games.nolimitholdem.round import Action
import tensorflow as tf
from rlcard.utils import print_card
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
    Logger,
    plot_curve,
)
from collections import namedtuple
from rlcard.agents.dqn_agent import DQNAgent

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done', 'legal_actions'])
# Set up the game environment
env = rlcard.make('no-limit-holdem')

# Create the human agent object
human_agent = HumanAgent(env.num_actions)

replay_memory_size = 20000
replay_memory_init_size = 1000
update_target_estimator_every = 1000
discount_factor = 0.99
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay_steps = 20000
batch_size = 32
num_actions = env.num_actions
state_shape=env.state_shape[0]
train_every = 1
mlp_layers = [128, 128]
learning_rate = 0.001
device = get_device()

# Create the DQN agent object
dqn_agent = DQNAgent(
    replay_memory_size=replay_memory_size,
    replay_memory_init_size=replay_memory_init_size,
    update_target_estimator_every=update_target_estimator_every,
    discount_factor=discount_factor,
    epsilon_start=epsilon_start,
    epsilon_end=epsilon_end,
    epsilon_decay_steps=epsilon_decay_steps,
    batch_size=batch_size,
    num_actions=num_actions,
    state_shape=state_shape,
    train_every=train_every,
    mlp_layers=mlp_layers,
    learning_rate=learning_rate,
    device=device
)


# Reset the environment
state, _ = env.reset()

env.set_agents([human_agent, dqn_agent])


while (True):
    print(">> Start a new game")

    trajectories, payoffs = env.run(is_training=False)
    # If the human does not take the final action, we need to
    # print other players action
    final_state = trajectories[0][-1]
    action_record = final_state['action_record']
    state = final_state['raw_obs']
    _action_list = []
    for i in range(1, len(action_record)+1):
        if action_record[-i][0] == state['current_player']:
            break
        _action_list.insert(0, action_record[-i])
    for pair in _action_list:
        print('>> Player', pair[0], 'chooses', pair[1])

    # Let's take a look at what the agent card is
    print('===============     Cards all Players    ===============')
    for hands in env.get_perfect_information()['hand_cards']:
        print("hola",num_actions)
        print_card(hands)

    print('===============     Result     ===============')
    if payoffs[0] > 0:
        print('You win {} chips!'.format(payoffs[0]))
    elif payoffs[0] == 0:
        print('It is a tie.')
    else:
        print('You lose {} chips!'.format(-payoffs[0]))
    print('')

    input("Press any key to continue...")
