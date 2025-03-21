class DummyScheduler:
    """
    Dummy LR Scheduler that supports standard methods like state_dict, load_state_dict, etc.,
    but does nothing to the optimizer or learning rates.
    """
    def __init__(self, optimizer, *args, **kwargs):
        """
        Initialize the DummyScheduler.

        Args:
            optimizer (Optimizer): Wrapped optimizer (required to match the API, not used).
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self.optimizer = optimizer
        self._state = {}

    def step(self, *args, **kwargs):
        """
        Dummy step function that does nothing.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        pass

    def state_dict(self):
        """
        Return the state of the scheduler as a dictionary.

        Returns:
            dict: A dictionary representing the scheduler's state.
        """
        return self._state

    def load_state_dict(self, state_dict):
        """
        Load the scheduler's state from a dictionary.

        Args:
            state_dict (dict): The state dictionary to load.
        """
        self._state.update(state_dict)

    def get_last_lr(self):
        """
        Get the last computed learning rate(s).

        Returns:
            list: A list of the last learning rates.
        """
        return [group['lr'] for group in self.optimizer.param_groups]