from typing import Iterator
from uuid import uuid4

from fastapi import FastAPI  # noqa
from pydantic.main import Model  # noqa

from rocket_learn.utils.experience_buffer import ExperienceBuffer
from rocket_learn.rollout_generator.base_rollout_generator import BaseRolloutGenerator


app = FastAPI(
    title="rocket-learn-api",
    version="0.1.0"
)

workers = {}


@app.post("/worker")
async def create_worker(name: str = "contributor"):
    uuid = uuid4()
    workers[uuid] = name
    return uuid

@app.get("/matchup")
async def get_matchup(mode: int):
    qualities = [...]

@app.post("/rollout")
async def rollout(obs_rew_probs: bytes, ):



class ApiRolloutGenerator(BaseRolloutGenerator):
    def generate_rollouts(self) -> Iterator[ExperienceBuffer]:
        pass

    def update_parameters(self, new_params):
        pass