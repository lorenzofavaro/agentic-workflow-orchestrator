class Budget:
    """
    Represents a budget for operations, tracking available USD and optional deadline in seconds.
    """
    def __init__(self, usd_left: float, deadline_s: float = None):
        """
        Initialize a Budget instance.
        Args:
            usd_left (float): Amount of USD left in the budget.
            deadline_s (float, optional): Deadline in seconds, if any.
        """
        self.usd_left = usd_left
        self.deadline_s = deadline_s

    def allow(self, cost: float, latency: float) -> bool:
        """
        Check if a given cost and latency are allowed by the current budget and deadline.
        Args:
            cost (float): The cost to check.
            latency (float): The expected latency in seconds.
        Returns:
            bool: True if the operation is allowed, False otherwise.
        """
        if self.usd_left < cost:
            return False
        if self.deadline_s is not None and self.deadline_s < latency:
            return False
        return True

    def charge(self, cost: float) -> None:
        """
        Deduct a cost from the available USD in the budget.
        Args:
            cost (float): The cost to deduct.
        """
        self.usd_left -= cost
