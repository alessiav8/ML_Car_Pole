import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Configuration parameters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
max_steps_per_episode = 100000
env = gym.make("CartPole-v1", render_mode="human")


gym.utils.seeding.np_random(seed)
eps = np.finfo(
    np.float32
).eps.item()  # Smallest number such that 1.0 + eps != 1.0num_inputs = 4
num_actions = 2
num_hidden = 128  # reti neurali ed è utilizzata per introdurre non linearità nei dati
num_inputs = 4  # spazio degli stati (angolo del palo, la velocità dell'angolo, la posizione del carrello e la velocità del carrello)

inputs = layers.Input(shape=(num_inputs,))
common = layers.Dense(num_hidden, activation="relu")(
    inputs
)  # condiviso da entrambi i rami dell'output.
action = layers.Dense(num_actions, activation="softmax")(
    common
)  # il primo ramo dell'output del modello
critic = layers.Dense(1)(
    common
)  # secondo ramo dell'output del modello, che rappresenta il critico

# Il primo ramo produce le probabilità delle azioni.
# Queste probabilità vengono calcolate utilizzando l'attivazione softmax,
# che converte un vettore di valori in un vettore di probabilità, consentendo al modello di
# selezionare un'azione tra due opzioni possibili.
#
# Il secondo ramo è il critico, che produce una singola stima delle ricompense future previste.
# Questo valore aiuta a valutare quanto una data azione sia buona o cattiva in base alle previsioni delle future ricompense.

model = keras.Model(inputs=inputs, outputs=[action, critic])

optimizer = keras.optimizers.legacy.Adam(learning_rate=0.01)
# L'ottimizzatore Adam è un algoritmo di ottimizzazione comunemente utilizzato in deep learning.
# Il tasso di apprendimento (learning_rate) è impostato su 0,01, che controlla la
# dimensione dei passi che l'ottimizzatore prende per aggiornare i pesi del modello durante l'addestramento.
huber_loss = keras.losses.Huber()
action_probs_history = (
    []
)  # memorizza i log delle probabilità delle azioni prese durante l'addestramento.
critic_value_history = []  # memorizza i valori previsti dal critico.
rewards_history = []  # memorizza le ricompense ricevute durante l'addestramento.
running_reward = 0
episode_count = 0

while True:  # Run until solved
    state = env.reset()[0]
    episode_reward = 0
    with tf.GradientTape() as tape:
        for timestep in range(1, max_steps_per_episode):
            state = np.array(state, dtype=np.float32)
            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, axis=0)

            # Predict action probabilities and estimated future rewards
            # from environment state
            action_probs, critic_value = model(state)
            critic_value_history.append(critic_value[0, 0])

            # Sample action from action probability distribution
            # action prob --> Actor
            action = np.random.choice(num_actions, p=np.squeeze(action_probs))
            action_probs_history.append(tf.math.log(action_probs[0, action]))

            # Apply the sampled action in our environment
            (
                next_state,
                reward,
                done,
                _,
                _,
            ) = env.step(action)
            rewards_history.append(reward)
            episode_reward += reward

            if done:
                break

            state = next_state

        # Update running reward to check condition for solving
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        # 0.05: Questo valore rappresenta il peso dato alla ricompensa totale dell'episodio corrente nel calcolo della media mobile
        # (1 - 0.05): Questo valore rappresenta il peso dato alla media mobile precedente
        # In sostanza, questa formula combina la ricompensa totale dell'episodio corrente con la media mobile delle ricompense
        # totali degli episodi precedenti. Questo approccio permette di ottenere una
        # stima più stabile delle prestazioni globali dell'agente e di monitorare il miglioramento nel tempo

        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for our critic
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()

        # Calculating loss values to update our network
        history = zip(action_probs_history, critic_value_history, returns)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up recieving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = ret - value
            actor_losses.append(-log_prob * diff)  # actor loss

            # The critic must be updated so that it predicts a better estimate of
            # the future rewards.
            critic_losses.append(
                huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )

        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Clear the loss and reward history
        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()

    # Log details
    episode_count += 1
    if episode_count % 10 == 0:
        template = "running reward: {:.2f} at episode {}"
        print(template.format(running_reward, episode_count))

    if running_reward > 195:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        break

env.close()
