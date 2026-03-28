from typing import Tuple

import torch


class WeatherForecast:
    def __init__(self, data_raw: list[list[float]]):
        """
        You are given a list of 10 weather measurements per day.
        Save the data as a PyTorch (num_days, 10) tensor,
        where the first dimension represents the day,
        and the second dimension represents the measurements.
        """
        self.data = torch.as_tensor(data_raw).view(-1, 10)

    def find_min_and_max_per_day(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find the max and min temperatures per day

        Returns:
            min_per_day: tensor of size (num_days,)
            max_per_day: tensor of size (num_days,)
        """
        min_per_day, _ = torch.min(self.data, dim=1)
        max_per_day, _ = torch.max(self.data, dim=1)
        return min_per_day, max_per_day 

    def find_the_largest_drop(self) -> torch.Tensor:
        """
        Find the largest change in day over day average temperature.
        This should be a negative number.

        Returns:
            tensor of a single value, the difference in temperature
        """
        avg_temps = torch.mean(self.data, dim=1)
        day_over_day_diff = avg_temps[1:] - avg_temps[:-1]
        # get largest drop (most negative)
        max_drop = torch.min(day_over_day_diff)
        return max_drop

    def find_the_most_extreme_day(self) -> torch.Tensor:
        """
        For each day, find the measurement that differs the most from the day's average temperature

        Returns:
            tensor with size (num_days,)
        """
        avg_temps = torch.mean(self.data, dim=1, keepdim=True)
        diffs = torch.abs(self.data - avg_temps)
        # return original measurement of the max diff for each day
        max_diff_indices = torch.argmax(diffs, dim=1)
        return self.data[torch.arange(self.data.size(0)), max_diff_indices]

    def max_last_k_days(self, k: int) -> torch.Tensor:
        """
        Find the maximum temperature over the last k days

        Returns:
            tensor of size (k,)
        """
        last_k_days = self.data[-k:]
        max_temps, _ = torch.max(last_k_days, dim=1)
        return max_temps

    def predict_temperature(self, k: int) -> torch.Tensor:
        """
        From the dataset, predict the temperature of the next day.
        The prediction will be the average of the temperatures over the past k days.

        Args:
            k: int, number of days to consider

        Returns:
            tensor of a single value, the predicted temperature
        """
        last_k_days = self.data[-k:]
        predicted_temp = torch.mean(last_k_days)
        return predicted_temp

    def what_day_is_this_from(self, t: torch.FloatTensor) -> torch.LongTensor:
        """
        You go on a stroll next to the weather station, where this data was collected.
        You find a phone with severe water damage.
        The only thing that you can see in the screen are the
        temperature reading of one full day, right before it broke.

        You want to figure out what day it broke.

        The dataset we have starts from Monday.
        Given a list of 10 temperature measurements, find the day in a week
        that the temperature is most likely measured on.

        We measure the difference using 'sum of absolute difference
        per measurement':
            d = |x1-t1| + |x2-t2| + ... + |x10-t10|

        Args:
            t: tensor of size (10,), temperature measurements

        Returns:
            tensor of a single value, the index of the closest data element
        """
        sum_diffs = torch.sum(torch.abs(self.data - t), dim=1)
        closest_day_index = torch.argmin(sum_diffs)
        return closest_day_index

        
        
