import numpy as np
from util.optimizer import AdamOptimizer
from attacks.base import Attack
import time

class CarliniWagnerAttack(Attack):
    """
    The class of the Carlini & Wagner attack.

    This attack is described in [1]_. This implementation is based on the 
    reference implemented by Carlini [2]_.
    
    References
    ----------
    .. [1] Carlini, Nicholas, and David Wagner. "Audio adversarial examples: 
           Targeted attacks on speech-to-text." 2018 IEEE Security and Privacy 
           Workshops (SPW). IEEE, 2018. 
    .. [2] https://github.com/carlini/audio_adversarial_examples 
    """

    def _attack_implementation(self, 
                adv_server, 
                learning_rate=10, 
                num_iterations=5000, 
                l2penalty=float('inf'),
                left_bound=-2000,
                right_bound=2000,
                verbose=True):
        original = adv_server.unperturbed
        delta = np.zeros_like(original)
        rescale = np.full(shape=original.shape[0], fill_value=1.0)
        optimizer = AdamOptimizer(original.shape)

        elapse_time = 0
        for i in range(num_iterations):
            start_time = time.time()
            # clip the perturbation
            apply_delta = np.clip(delta, left_bound, right_bound) * rescale[:, np.newaxis]
            new_input = original + apply_delta
            #noise = np.random.normal(scale=2, size=original.shape)
            pass_in = np.clip(new_input, a_min=-2**15, a_max=2**15-1).astype(np.int16)
            pred_trans, g_loss_input, g_distance_input, is_adversarial, is_best, distance = adv_server.post_new_data(pass_in)

            if not np.isinf(l2penalty):
                total_gradient = g_distance_input + l2penalty * g_loss_input
            else:
                total_gradient = g_loss_input

            # update perturbation
            delta += optimizer(np.sign(total_gradient), learning_rate)
            elapse_time += time.time() - start_time
            # print out some debug information every 10 iterations
            if verbose and i % 10 == 0:
                is_advs = ['Yes' if is_adv else 'No' for is_adv in is_adversarial]
                print('-' * 10 + 'Iteration: {}'.format(i) + ', elapse time: {}'.format(elapse_time)+ '-' * 10)
                print('{:^100s}{:^12s}{:^6s}'.format('Prediction', 'Distance', 'Adv'))
                print('{:100s}{:^12s}{:^6s}'.format('-'*98, '-'*10, '-'*6))
                res = ['{:100s}{:^.4E}{:^6s}'.format(pred[0:96], di, is_adv) for pred, di, is_adv in zip(pred_trans, distance, is_advs)]
                print('\n'.join(res))
                elapse_time = 0

            if i % 10 == 0 and np.any(is_adversarial):
                new_scale = np.minimum(np.max(np.abs(delta), axis=-1) / right_bound, rescale)
                rescale[is_adversarial] = new_scale[is_adversarial] * 0.8
