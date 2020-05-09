import numpy as np
from util.optimizer import AdamOptimizer
from attacks.base import Attack
import time

class YaoCarliniAttack(Attack):
    """
    The class of the Yao & Carlini attack. This is a two-stage attack.

    This attack is described in [1]_. This implementation is based on the 
    reference implemented by Qin Yao [2]_.
    
    References
    ----------
    .. [1] Qin, Yao, et al. "Imperceptible, robust, and targeted adversarial examples for automatic 
           speech recognition." PMLR (2019). 
    .. [2] https://github.com/tensorflow/cleverhans/tree/master/examples/adversarial_asr
    """

    def _attack_implementation(self, 
                adv_server, 
                lr_stage1=100,
                lr_stage2=0.1,
                num_iter_stage1=1000,
                num_iter_stage2=4000,
                left_bound=-2000,
                right_bound=2000,
                l2penalty=0.05,
                verbose=True):
        original = adv_server.unperturbed
        delta = np.zeros_like(original)
        rescale = np.full(shape=original.shape[0], fill_value=1.0)
        alpha   = np.full(shape=original.shape[0], fill_value=l2penalty)

        optimizer1 = AdamOptimizer(original.shape)
        optimizer2 = AdamOptimizer(original.shape)

        elapse_time = 0
        for i in range(num_iter_stage1 + num_iter_stage2):
            start_time = time.time()
            # clip the perturbation
            apply_delta = np.clip(delta, left_bound, right_bound) * rescale[:, np.newaxis]
            new_input = original + apply_delta
            pass_in = np.clip(new_input, a_min=-2**15, a_max=2**15-1).astype(np.int16)
            pred_trans, g_loss_input, g_distance_input, is_adversarial, _, distance = adv_server.post_new_data(pass_in)

            if i < num_iter_stage1:
                ptt = optimizer1(np.sign(g_loss_input), lr_stage1)
            else:
                ptt = optimizer2(g_loss_input + g_distance_input * alpha[:, np.newaxis], lr_stage2)

            delta += ptt 
            elapse_time += time.time() - start_time
            # print out some debug information every 10 iterations
            if verbose and i % 20 == 0:
                is_advs = ['Yes' if is_adv else 'No' for is_adv in is_adversarial]
                print('-' * 50 + 'Iteration: {}, Stage: {}, Elapse time: {}'.format(i, 1 if i < num_iter_stage1 else 2, elapse_time) + '-' * 50)
                print('{:^100s}{:^12s}{:6s}'.format('Prediction', 'Distance', 'Adv'))
                print('{:100s}{:^12s}{:^6s}'.format('-'*98, '-'*10, '-'*6))
                res = ['{:100s}{:^.4E}{:^6s}'.format(pred[0:96], di, is_adv) for pred, di, is_adv in zip(pred_trans, distance, is_advs)]
                print('\n'.join(res))
                elapse_time = 0

            if i < num_iter_stage1 and i % 10 == 0 and np.any(is_adversarial):
                new_scale = np.minimum(np.max(np.abs(delta), axis=-1) / right_bound, rescale)
                rescale[is_adversarial] = new_scale[is_adversarial] * 0.8
           
            if i >= num_iter_stage1 and i % 20 == 0:
                alpha[is_adversarial] *= 1.2

            if i >= num_iter_stage1 and i % 50 == 0:
                alpha[np.logical_not(is_adversarial)] *= 0.8
