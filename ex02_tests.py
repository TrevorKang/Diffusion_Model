import unittest
import torch
import matplotlib.pyplot as plt
from numpy.testing import assert_almost_equal

from ex02_diffusion_origin import Diffusion, linear_beta_schedule


class TestDiffusionModel(unittest.TestCase):
    def setUp(self):
        self.test_values = torch.load("ex02_test_values.pt")
        # expected_output = self.test_values['q_sample']['expected_output']
        # self.expected_output = expected_output.reshape(expected_output.shape[0], -1).numpy()
        # # visualize the expected output
        # plt.imshow(expected_output[0, 0, :, :], cmap="gray")
        # plt.show()
        self.scheduler = lambda x: linear_beta_schedule(0.001, 0.02, x)
        self.img_size = 32

    def test_q_sample(self):
        local_values = self.test_values["q_sample"]
        print(local_values["noise"].shape)
        # visualize the expected output
        # plt.imshow(local_values["expected_output"][0, 0, :, :])
        # plt.show()
        diffusor = Diffusion(timesteps=local_values["timesteps"],
                             get_noise_schedule=self.scheduler, img_size=self.img_size)

        output = diffusor.q_sample(x_zero=local_values["x_zero"].cuda(),
                                   t=local_values["t"].cuda(), noise=local_values["noise"].cuda())
        assert_almost_equal(local_values["expected_output"].numpy(), output.cpu().numpy(), decimal=5)


def main():
    unittest.main()


if __name__ == "__main__":
    main()
