"""Hard task grader."""

class HardGrader:
    def grade(self, env, *args, **kwargs) -> float:
        score = env.get_grader_score()
        return max(0.01, min(0.99, float(score)))
