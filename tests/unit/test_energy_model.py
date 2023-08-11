import pytest

from aml.models.energy_models import BPNN, MACE, Allegro, GemNetT, NequIP, PaiNN, SchNet

all_models = [BPNN, MACE, Allegro, GemNetT, NequIP, PaiNN, SchNet]


@pytest.mark.parametrize("model", all_models)
def test_molecule(model, water_molecule):
    kwargs = {}
    if model in (MACE, NequIP, Allegro):
        kwargs["avg_num_neighbors"] = 4
    model = model(["H", "O"], **kwargs)
    energy = model(water_molecule)
    assert energy.shape == (1,)


@pytest.mark.parametrize("model", all_models)
def test_batch(model, batch):
    kwargs = {}
    if model in (MACE, NequIP, Allegro):
        kwargs["avg_num_neighbors"] = 15
    model = model(["Si", "Ge", "Pt", "Ni"], **kwargs)
    energy = model(batch)
    assert energy.shape == (2,)
