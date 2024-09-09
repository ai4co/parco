import torch
from logging import getLogger

from FFSPEnv import FFSPEnv as Env
from FFSPModel import FFSPModel

from torch.optim import AdamW as Optimizer
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR, ReduceLROnPlateau, CyclicLR, CosineAnnealingLR

import wandb

from utils.utils import *
from FFSProblemDef import load_problems_from_file, get_random_problems

import numpy as np
#
scheduler_map = {
    "multistep": MultiStepLR,
    "exponential": ExponentialLR,
    "cyclic": CyclicLR,
    "plateau": ReduceLROnPlateau,
    "cos": CosineAnnealingLR
}



class FFSPTrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params,
                 tester_params):

        # save arguments
        torch.manual_seed(env_params["seed"])
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params
        self.tester_params = tester_params
        self.grad_accumulation_steps = trainer_params["accumulation_steps"]
        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        self.wandb = wandb.init(
            project="parco-ffsp",
            tags=[
                f"jobs:{env_params['job_cnt']}",
                f"machines:{env_params['ma_cnt_str']}"
            ],
            config={
                "model": dict(model_params),
                "optimizer": dict(optimizer_params),
                "train": dict(trainer_params)
            }
        )

        # cuda
        use_cuda = self.trainer_params['use_cuda'] and torch.cuda.is_available()
        if use_cuda:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')


        self.model = FFSPModel(**self.model_params)
        self.env = Env(**self.env_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        Scheduler = scheduler_map[self.optimizer_params['scheduler']["class"]]
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler']["kwargs"])

        # restore
        self.start_epoch = 1


        # utility
        self.time_estimator = TimeEstimator()

        # Load all problems
        self.logger.info(" *** Loading Saved Problems *** ")
        saved_problem_folder = self.tester_params['saved_problem_folder']
        saved_problem_filename = self.tester_params['saved_problem_filename']
        filename = os.path.join(saved_problem_folder, saved_problem_filename)
        try:
            self.ALL_problems_INT_list = load_problems_from_file(filename, device=device)
        except:
            self.ALL_problems_INT_list = get_random_problems(
                self.tester_params["problem_count"],
                self.env_params["machine_cnt_list"],
                self.env_params["job_cnt"],
                self.env_params["process_time_params"]
            )

        self.logger.info("Done. ")


    def run(self):
        """
        Run training for multiple epochs
        """

        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')



            # Train
            train_score, train_loss, steps = self._train_one_epoch(epoch)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)
            self.result_log.append('steps', epoch, steps)
            # LR Decay
            self.scheduler.step()

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']

            if epoch > 1:  # save latest images, every epoch
                self.logger.info("Saving log_image")
                image_prefix = '{}/latest'.format(self.result_folder)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                               self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                               self.result_log, labels=['train_loss'])

            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

            if all_done or (epoch % img_save_interval) == 0:
                image_prefix = '{}/img/checkpoint-{}'.format(self.result_folder, epoch)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                               self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                               self.result_log, labels=['train_loss'])

            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)



    def _train_one_epoch(self, epoch):

        score_AM = AverageMeter()
        loss_AM = AverageMeter()
        steps_AM = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0

        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            avg_score, avg_loss, steps_mean = self._train_one_batch(batch_size, self.grad_accumulation_steps)
            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)
            steps_AM.update(steps_mean, batch_size)

            episode += batch_size

            self.logger.info(
                'Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}, Steps: {}'
                .format(
                    epoch, episode, train_num_episode, 
                    100. * episode / train_num_episode, 
                    score_AM.avg, loss_AM.avg, steps_AM.avg
                )
            )
            
        self.logger.info("skip ratio: {}".format(np.mean(self.env.skip_ratio)))

        self.wandb.log({
            "score": score_AM.avg,
            "loss": loss_AM.avg,
            "steps": steps_AM.avg,
        })
        return score_AM.avg, loss_AM.avg, steps_AM.avg

    def _train_one_batch(self, batch_size, accumulation_steps=1):
        
        # Prep
        ###############################################
        self.model.train()
        mini_batch_size = batch_size // accumulation_steps
        self.optimizer.zero_grad()
        score_mean = 0
        steps_mean = 0
        for _ in range(accumulation_steps):
            self.env.load_problems(mini_batch_size)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)
            # shape: (batch, pomo, 0~makespan)
            prob_list = torch.zeros(size=(mini_batch_size, self.env.pomo_size, 0))
            # Rollout
            state, reward, done = self.env.pre_step()

            steps = 0
            while not done:
                jobs, machines, prob = self.model(state)
                # shape: (batch, pomo)
                state, reward, done = self.env.step(jobs, machines)
                prob_list = torch.cat((prob_list, prob), dim=-1)
                steps += 1
            # LEARNING
            ###############################################
            advantage = reward - reward.float().mean(dim=1, keepdims=True)
            # shape: (batch, pomo)
            log_prob = prob_list.log().sum(dim=2)
            # size = (batch, pomo)
            loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
            # shape: (batch, pomo)

            loss_mean = loss.mean() / accumulation_steps
            loss_mean.backward()

            # Score
            ###############################################
            max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
            score_mean += -(max_pomo_reward.float().mean().item() / accumulation_steps)  # negative sign to make positive value
            steps_mean += steps / accumulation_steps

        for group in self.optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(group['params'], self.trainer_params["max_grad_norm"])

        # Step & Return
        ###############################################
        self.optimizer.step()
        self.model.zero_grad()

        return score_mean, loss_mean.item(), steps_mean


    def _train_one_batch_self_labeling(self, batch_size):
        
        # Prep
        ###############################################
        self.model.train()
        self.env.load_problems(batch_size)
        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state)

        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
        # shape: (batch, pomo, 0~makespan)

        # Rollout
        ###############################################
        state, reward, done = self.env.pre_step()

        while not done:

            jobs, machines, prob = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(jobs, machines)

            prob_list = torch.cat((prob_list, prob), dim=-1)

        # LEARNING
        ###############################################
        max, argmax = reward.float().max(dim=1)
        # shape: (batch, pomo)
        log_prob = prob_list.log().sum(dim=2)
        # size = (batch, pomo)
        loss = -log_prob.gather(1, argmax.unsqueeze(1)).mean()


        # Score
        ###############################################
        score_mean = -max.float().mean()  # negative sign to make positive value

        # Step & Return
        ###############################################
        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()
        return score_mean.item(), loss.item()


    def eval(self):

        # save_solution = self.tester_params['save_solution']['enable']
        # solution_list = []

        self.time_estimator.reset()

        score_AM = AverageMeter()
        aug_score_AM = AverageMeter()

        test_num_episode = self.tester_params['problem_count']
        episode = 0

        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            problems_INT_list = []
            for stage_idx in range(self.env.stage_cnt):
                problems_INT_list.append(self.ALL_problems_INT_list[stage_idx][episode:episode+batch_size])

            score, aug_score = self._test_one_batch(problems_INT_list)

            score_AM.update(score, batch_size)
            aug_score_AM.update(aug_score, batch_size)

            episode += batch_size

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, aug_score:{:.3f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score, aug_score))

            all_done = (episode == test_num_episode)

            if all_done:
                self.logger.info(" *** Test Done *** ")
                self.logger.info(" NO-AUG SCORE: {:.4f} ".format(score_AM.avg))
                self.logger.info(" AUGMENTATION SCORE: {:.4f} ".format(aug_score_AM.avg))

                self.wandb.log({
                    "test_score": score_AM.avg,
                    "test_score_aug": aug_score_AM.avg,
                })

    def _test_one_batch(self, problems_INT_list):

        batch_size = problems_INT_list[0].size(0)

        # Augmentation
        ###############################################
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
            batch_size = aug_factor*batch_size
            for stage_idx in range(self.env.stage_cnt):
                problems_INT_list[stage_idx] = problems_INT_list[stage_idx].repeat(aug_factor, 1, 1)
                # shape: (batch*aug_factor, job_cnt, machine_cnt)
        else:
            aug_factor = 1

        # Ready
        ###############################################
        self.model.eval()
        with torch.no_grad():
            self.env.load_problems_manual(problems_INT_list)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

            # POMO Rollout
            ###############################################
            state, reward, done = self.env.pre_step()
            while not done:
                jobs, machines, _ = self.model(state)
                # shape: (batch, pomo)
                state, reward, done = self.env.step(jobs, machines)

            # Return
            ###############################################
            batch_size = batch_size//aug_factor
            aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
            # shape: (augmentation, batch, pomo)

            max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
            # shape: (augmentation, batch)
            no_aug_score = -max_pomo_reward[0, :].float().mean()  # negative sign to make positive value

            max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
            # shape: (batch,)
            aug_score = -max_aug_pomo_reward.float().mean()  # negative sign to make positive value

            return no_aug_score.item(), aug_score.item()