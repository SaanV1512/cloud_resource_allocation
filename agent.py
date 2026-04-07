import random

class AdaptiveAgent:
    def __init__(self):
        # Initial thresholds
        self.u_high = 0.8
        self.u_mid  = 0.7
        self.u_low  = 0.3
        self.Q_high = 5

        # Memory
        self.prev_state = None
        self.last_action = None

        # Exploration
        self.epsilon = 0.05

    def act(self, state):
        d_t = state["demand"]
        q_t = state["queue"]
        u_t = state["utilization"]

        # previous values
        if self.prev_state:
            d_prev = self.prev_state["demand"]
            q_prev = self.prev_state["queue"]
        else:
            d_prev = d_t
            q_prev = q_t

        # exploration
        if random.random() < self.epsilon:
            action = random.choice([-1, 0, 1])
        else:
            # backlog priority
            if q_t > self.Q_high:
                action = 1

            # trend
            elif (d_t > d_prev) and (u_t > self.u_mid):
                action = 1

            # overload
            elif (u_t > self.u_high) and (q_t > q_prev):
                action = 1

            # scale down
            elif (u_t < self.u_low) and (q_t == 0):
                action = -1

            else:
                action = 0

        # stability rule
        if self.last_action == 1 and action == 1:
            action = 0

        # update memory
        self.prev_state = state
        self.last_action = action

        return action

    def learn(self, reward, info):
        """
        Adjust thresholds based on reward feedback
        info should contain:
        - latency
        - cost
        - instability
        - sla_violations
        """

        latency = info.get("latency", 0)
        cost = info.get("cost", 0)
        instability = info.get("instability", 0)
        sla = info.get("sla_violations", 0)

        # High latency → react earlier
        if latency > 0.7:
            self.u_high = max(0.6, self.u_high - 0.02)
            self.Q_high = max(2, self.Q_high - 1)

        # High cost → scale down earlier
        if cost > 0.7:
            self.u_low = min(0.5, self.u_low + 0.02)

        # SLA violations → be aggressive
        if sla > 0:
            self.u_high = max(0.6, self.u_high - 0.03)
            self.Q_high = max(2, self.Q_high - 1)

        # Instability → reduce sensitivity
        if instability > 0.5:
            self.u_high = min(0.9, self.u_high + 0.01)
#how to use in environment:
# agent = AdaptiveAgent()

# state = env.reset()
# done = False

# while not done:
#     action = agent.act(state)

#     next_state, reward, done, info = env.step(action)

#     agent.learn(reward, info)

#     state = next_state