import multiprocessing as mp
import numpy as np
import logging
import os
import sys
import dcppg as network
import tensorflow.compat.v1 as tf
import gym

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

S_DIM = 4
A_DIM = 2
ACTOR_LR_RATE = 1e-4
NUM_AGENTS = 8
TRAIN_SEQ_LEN = 500  # take as a train batch
TRAIN_EPOCH = 500000
MODEL_SAVE_INTERVAL = 50
RANDOM_SEED = 42
SUMMARY_DIR = './dcppg'
LOG_FILE = SUMMARY_DIR + '/log'

# create result directory
if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)

NN_MODEL = None    

def central_agent(net_params_queues, exp_queues):

    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS
    tf_config=tf.ConfigProto(intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=1)
    with tf.Session(config = tf_config) as sess, open(LOG_FILE + '_test.txt', 'w') as test_log_file:
        summary_ops, summary_vars = build_summaries()

        actor = network.Network(sess,
                state_dim=S_DIM, action_dim=A_DIM,
                learning_rate=ACTOR_LR_RATE)

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)  # training monitor
        saver = tf.train.Saver(max_to_keep=1000)  # save neural net parameters

        # restore neural net parameters
        nn_model = NN_MODEL
        if nn_model is not None:  # nn_model is the path to file
            saver.restore(sess, nn_model)
            print("Model restored.")
        
        # while True:  # assemble experiences from agents, compute the gradients
        for epoch in range(TRAIN_EPOCH):
            # synchronize the network parameters of work agent
            actor_net_params = actor.get_network_params()
            for i in range(NUM_AGENTS):
                net_params_queues[i].put(actor_net_params)

            s, a, p, g, r = [], [], [], [], []
            for i in range(NUM_AGENTS):
                s_, a_, p_, g_, r_ = exp_queues[i].get()
                s += s_
                a += a_
                p += p_
                g += g_
                r.append(np.sum(r_))

            actor.train(s, a, p, g, epoch)

            if epoch % MODEL_SAVE_INTERVAL == 0:
                # Save the neural net parameters to disk.
                save_path = saver.save(sess, SUMMARY_DIR + "/nn_model_ep_" +
                                       str(epoch) + ".ckpt")
                
                test_log_file.write(str(epoch) + '\t' +
                   str(np.mean(r)) + '\n')
                test_log_file.flush()

            summary_str = sess.run(summary_ops, feed_dict={
                summary_vars[0]: np.mean(r)
            })
            writer.add_summary(summary_str, epoch)
            writer.flush()

def agent(agent_id, net_params_queue, exp_queue):
    env = gym.make("CartPole-v0")
    env.force_mag = 100.0

    with tf.Session() as sess:
        actor = network.Network(sess,
                                state_dim=S_DIM, action_dim=A_DIM,
                                learning_rate=ACTOR_LR_RATE)

        # initial synchronization of the network parameters from the coordinator
        actor_net_params = net_params_queue.get()
        actor.set_network_params(actor_net_params)

        time_stamp = 0

        for epoch in range(TRAIN_EPOCH):
            obs, info = env.reset()
            s_batch, a_batch, p_batch, r_batch = [], [], [], []
            for step in range(TRAIN_SEQ_LEN):
                s_batch.append(obs)
                
                action_prob = actor.predict(obs)

                noise = np.random.gumbel(size=len(action_prob))
                action = np.argmax(np.log(action_prob) + noise)
                obs, rew, done, truncated, info = env.step(action)

                action_vec = np.zeros(A_DIM)
                action_vec[action] = 1

                a_batch.append(action_vec)

                r_batch.append(rew)
                p_batch.append(action_prob)

                if done:
                    break

            v_batch = actor.compute_v(s_batch, r_batch, done)

            exp_queue.put([s_batch, a_batch, p_batch, v_batch, r_batch])

            actor_net_params = net_params_queue.get()
            actor.set_network_params(actor_net_params)

def build_summaries():
    eps_total_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", eps_total_reward)

    summary_vars = [eps_total_reward]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

def main():

    np.random.seed(RANDOM_SEED)

    # inter-process communication queues
    net_params_queues = []
    exp_queues = []
    for i in range(NUM_AGENTS):
        net_params_queues.append(mp.Queue(1))
        exp_queues.append(mp.Queue(1))

    # create a coordinator and multiple agent processes
    # (note: threading is not desirable due to python GIL)
    coordinator = mp.Process(target=central_agent,
                             args=(net_params_queues, exp_queues))
    coordinator.start()

    agents = []
    for i in range(NUM_AGENTS):
        agents.append(mp.Process(target=agent,
                                 args=(i,
                                       net_params_queues[i],
                                       exp_queues[i])))
    for i in range(NUM_AGENTS):
        agents[i].start()

    # wait unit training is done
    coordinator.join()


if __name__ == '__main__':
    main()
