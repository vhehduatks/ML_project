from snake_game import SnakeGame
from agent import Agent
from Plot import plot

if __name__ == '__main__':
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGame()

    #game_loop
    while True:
        # get state
        state_old = game.get_state()

        # get move
        action = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(action)
        state_new = game.get_state()

        # train short memory
        agent.train_short_memory(state_old, action, reward, state_new, done)

        # remember
        agent.remember(state_old, action, reward, state_new, done)

        if done:#게임이 끝났을때
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)