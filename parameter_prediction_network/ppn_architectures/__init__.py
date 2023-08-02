from enum import Enum


class PPNArchitectures(Enum):
    johnson = 1
    johnson_no_batch_norm = 2
    johnson_instance_norm = 3
    unet_small_no_batch_norm = 4
    unet_small_instance_norm = 5
    unet_large_no_batch_norm = 6
    unet_large_instance_norm = 7

    def __str__(self):
        return self.name

    @staticmethod
    def to_log_name(arch: str):
        first_part_letters = [name_part[0] for name_part in str(PPNArchitectures[arch]).split('_')]
        return ''.join(first_part_letters)
