import time
import random
import numpy as np
import numpy.random
from parameters import args
import torch
import torch.optim as optim
from env.env_batch_het import JsspN5, BatchGraph
from model.actor_hetGAT_lab import Actor
from env.generateJSP import uni_instance_gen
from pathlib import Path


class RL2S4JSSP:
    def __init__(self):
        self.env_training = JsspN5(n_job=args.j, n_mch=args.m, low=args.l, high=args.h, reward_type=args.reward_type)
        self.env_validation = JsspN5(n_job=args.j, n_mch=args.m, low=args.l, high=args.h, reward_type=args.reward_type)
        self.eps = np.finfo(np.float32).eps.item()
        validation_data_path = Path('./validation_data/validation_instance_{}x{}[{},{}].npy'.format(args.j, args.m, args.l, args.h))
        if validation_data_path.is_file():
            self.validation_data = np.load('./validation_data/validation_instance_{}x{}[{},{}].npy'.format(args.j, args.m, args.l, args.h))
        else:
            print('No validation data for {}x{}[{},{}], generating new one.'.format(args.j, args.m, args.l, args.h))
            self.validation_data = np.array([uni_instance_gen(n_j=args.j, n_m=args.m, low=args.l, high=args.h) for _ in range(args.batch_size)])
            np.save('./validation_data/validation_instance_{}x{}[{},{}].npy'.format(args.j, args.m, args.l, args.h), self.validation_data)
        self.incumbent_validation_result = np.inf
        self.current_validation_result = np.inf

        if args.embedding_type == 'gin':
            self.dghan_param_for_saved_model = '{NAN}'
        elif args.embedding_type == 'dghan' or args.embedding_type == 'gin+dghan':
            self.dghan_param_for_saved_model = '{}_{}'.format(args.heads, args.drop_out)
        else:
            raise Exception('embedding_type should be one of "gin", "dghan", or "gin+dghan".')


    def learn(self, rewards, log_probs, dones, optimizer):
        R = torch.zeros_like(rewards[0], dtype=torch.float, device=rewards[0].device)
        returns = []
        for r in rewards[::-1]:
            R = r + args.gamma * R
            returns.insert(0, R)
        returns = torch.cat(returns, dim=-1)
        dones = torch.cat(dones, dim=-1)
        log_probs = torch.cat(log_probs, dim=-1)

        losses = []
        for b in range(returns.shape[0]):
            masked_R = torch.masked_select(returns[b], ~dones[b])
            masked_R = (masked_R - masked_R.mean()) / (torch.std(masked_R, unbiased=False) + self.eps)
            masked_log_prob = torch.masked_select(log_probs[b], ~dones[b])
            loss = (- masked_log_prob * masked_R).sum()
            losses.append(loss)

        optimizer.zero_grad()
        mean_loss = torch.stack(losses).mean()
        mean_loss.backward()
        optimizer.step()


    def validation(self, policy, dev):

        # fixed seed for fair validation: validation improve not because of different critical path
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)

        # validating...
        validation_start = time.time()
        validation_batch_data = BatchGraph()
        states_val, feasible_actions_val, dones_val = self.env_validation.reset(instances=self.validation_data,
                                                                                init_type=args.init_type,
                                                                                device=dev)
        while self.env_validation.itr < args.transit:
            validation_batch_data.wrapper(*states_val)
            actions_val, log_ps_val = policy(validation_batch_data, feasible_actions_val)
            states_val, rewards_vall, feasible_actions_val, dones_val = self.env_validation.step(actions_val, dev)
        states_val, rewards_vall, feasible_actions_val, dones_val, actions_val, log_ps_val = None, None, None, None, None, None
        validation_batch_data.clean()
        validation_result1 = self.env_validation.incumbent_objs.mean().cpu().item()
        validation_result2 = self.env_validation.current_objs.mean().cpu().item()
        # saving model based on validation results
        if validation_result1 < self.incumbent_validation_result:
            print('Find better model w.r.t incumbent objs, saving model...')
            torch.save(policy.state_dict(),
                       './saved_model/incumbent_'  # saved model type
                       '{}x{}[{},{}]_{}_{}_{}_'  # env parameters
                       '{}_{}_{}_{}_{}_'  # model parameters
                       '{}_{}_{}_{}_{}_{}'  # training parameters
                       '.pth'
                       .format(args.j, args.m, args.l, args.h, args.init_type, args.reward_type, args.gamma,
                               args.hidden_dim, args.embedding_layer, args.policy_layer, args.embedding_type, self.dghan_param_for_saved_model,
                               args.lr, args.steps_learn, args.transit, args.batch_size, args.episodes,
                               args.step_validation))
            self.incumbent_validation_result = validation_result1
        if validation_result2 < self.current_validation_result:
            print('Find better model w.r.t final step objs, saving model...')
            torch.save(policy.state_dict(),
                       './saved_model/last-step_'  # saved model type
                       '{}x{}[{},{}]_{}_{}_{}_'  # env parameters
                       '{}_{}_{}_{}_{}_'  # model parameters
                       '{}_{}_{}_{}_{}_{}'  # training parameters
                       '.pth'
                       .format(args.j, args.m, args.l, args.h, args.init_type, args.reward_type, args.gamma,
                               args.hidden_dim, args.embedding_layer, args.policy_layer, args.embedding_type, self.dghan_param_for_saved_model,
                               args.lr, args.steps_learn, args.transit, args.batch_size, args.episodes,
                               args.step_validation))
            self.current_validation_result = validation_result2

        validation_end = time.time()

        print('Incumbent objs and final step objs for validation are: {:.2f}  {:.2f}'.format(validation_result1, validation_result2),
              'validation takes:{:.2f}'.format(validation_end - validation_start))

        return validation_result1, validation_result2


    def train(self):

            dev = 'cuda' if torch.cuda.is_available() else 'cpu'
            init = args.init_type

            policy = Actor(in_dim=3,
                           hidden_dim=args.hidden_dim,
                           embedding_l=args.embedding_layer,
                           policy_l=args.policy_layer,
                           embedding_type=args.embedding_type,
                           heads=args.heads,
                           dropout=args.drop_out).to(dev)

            optimizer = optim.Adam(policy.parameters(), lr=args.lr)

            batch_data = BatchGraph()
            log = []
            validation_log = []

            print()
            for batch_i in range(1, args.episodes // args.batch_size + 1):

                t1 = time.time()

                random.seed(batch_i)
                np.random.seed(batch_i)
                torch.manual_seed(batch_i)

                instances = np.array([uni_instance_gen(args.j, args.m, args.l, args.h) for _ in range(args.batch_size)])
                print(instances)
                states, feasible_actions, dones = self.env_training.reset(instances=instances, init_type=init, device=dev)

                reward_log = []
                rewards_buffer = []
                log_probs_buffer = []
                dones_buffer = [dones]

                while self.env_training.itr < args.transit:
                    batch_data.wrapper(*states)
                    actions, log_ps = policy(batch_data, feasible_actions)
                    states, rewards, feasible_actions, dones = self.env_training.step(actions, dev)

                    # store training data
                    rewards_buffer.append(rewards)
                    log_probs_buffer.append(log_ps)
                    dones_buffer.append(dones)

                    # logging reward...
                    # reward_log.append(rewards)

                    if self.env_training.itr % args.steps_learn == 0:
                        # training...
                        self.learn(rewards_buffer, log_probs_buffer, dones_buffer[:-1], optimizer)
                        # clean training data
                        rewards_buffer = []
                        log_probs_buffer = []
                        dones_buffer = [dones]

                # learn(rewards_buffer, log_probs_buffer, dones_buffer[:-1])  # old-school training scheme

                t2 = time.time()
                print('Batch {} training takes: {:.2f}'.format(batch_i, t2 - t1),
                      'Mean Performance: {:.2f}'.format(self.env_training.current_objs.cpu().mean().item()))
                log.append(self.env_training.current_objs.mean().cpu().item())

                # start validation and saving model & logs...
                if batch_i % args.step_validation == 0:

                    validation_result1, validation_result2 = self.validation(policy, dev)
                    validation_log.append([validation_result1, validation_result2])

                    # saving log
                    np.save('./log/training_log_'
                            '{}x{}[{},{}]_{}_{}_{}_'  # env parameters
                            '{}_{}_{}_{}_{}_'  # model parameters
                            '{}_{}_{}_{}_{}_{}.npy'  # training parameters
                            .format(args.j, args.m, args.l, args.h, args.init_type, args.reward_type, args.gamma,
                                    args.hidden_dim, args.embedding_layer, args.policy_layer, args.embedding_type, self.dghan_param_for_saved_model,
                                    args.lr, args.steps_learn, args.transit, args.batch_size, args.episodes,
                                    args.step_validation),
                            np.array(log))
                    np.save('./log/validation_log_'
                            '{}x{}[{},{}]_{}_{}_{}_'  # env parameters
                            '{}_{}_{}_{}_{}_'  # model parameters
                            '{}_{}_{}_{}_{}_{}.npy'  # training parameters
                            .format(args.j, args.m, args.l, args.h, args.init_type, args.reward_type, args.gamma,
                                    args.hidden_dim, args.embedding_layer, args.policy_layer, args.embedding_type, self.dghan_param_for_saved_model,
                                    args.lr, args.steps_learn, args.transit, args.batch_size, args.episodes,
                                    args.step_validation),
                            np.array(validation_log))



if __name__ == '__main__':
    agent = RL2S4JSSP()
    agent.train()
