from rlgym.utils.gamestates import GameState

from rocket_learn.utils.util import encode_gamestate


def test_encode_gamestate():
    initial = [0.9014944386070818, 0.7437496717058292, 0.11523679140045551]
    boost_pad = [0.003111401210689868, 0.5021335460386673, 0.8533789339552476, 0.09714999894304932, 0.7143390721018188,
                 0.904023562293331, 0.16265660155305572, 0.571626944750347, 0.24778384184792368, 0.9091309632722162,
                 0.6592361326653748, 0.791795318386487, 0.2724695390498557, 0.32326792073286925, 0.167624621608909,
                 0.8361758804667154, 0.8746872111461466, 0.7678519252038587, 0.5267171401875672, 0.5168564049955375,
                 0.8574124397586926, 0.4390338879520339, 0.7161976230618752, 0.9423951164291233, 0.4252098997873843,
                 0.6348407562452576, 0.7996163381861632, 0.9607027302407571, 0.29866522048265587, 0.030547111448842923,
                 0.6698791835691189, 0.22825689183720432, 0.31471833657100423, 0.137017403696158]
    ball_state = [0.27696337802234927, 0.901607574537501, 0.9162636151172863, 0.41581528176420535, 0.3802807190091779,
                  0.2828253821061679, 0.9477133581539083, 0.7817742167252288, 0.06521346143339035, 0.07321247932474795,
                  0.35884491811630737, 0.22579710962370836, 0.7108677439023905, 0.21351596462413647,
                  0.14567335233924572, 0.31729176772958034, 0.011272366424840863, 0.8338832290060092]
    player_info = [0.1178052811634831, 0.7593878303755066, 0.3792401241859009, 0.0664535399729691, 0.9830587561000144,
                   0.2368589444233229, 0.146776662484974, 0.1874666469160684, 0.9514036150101033, 0.4132994067611899,
                   0.15133628109361186, 0.48387899467955564, 0.047234692358380626, 0.2308392882416661,
                   0.4935608251648853, 0.9113517608715533, 0.01738674352128089, 0.7335486816964638, 0.5818931065950903,
                   0.6313772865665637, 0.9449745140714203, 0.02902726416928747, 0.11522208711681747,
                   0.24542619510094865, 0.29808562243890857, 0.6950708345594185, 0.524042067861269, 0.48254636506229165,
                   0.28527828522822873, 0.9117533992528384, 0.45416210041176064, 0.47544287360060866,
                   0.08661855564023346, 0.3795672723524046, 0.9262328865485365, 0.8992876308061273, 0.47634356601984007,
                   0.19171424900276124]
    player_car_state = [0.5715392654342994, 0.9559166840971312, 0.25425424051431444, 0.6333342900076249,
                        0.40481192086972473, 0.5077252706866591, 0.4583819597091836, 0.4023090090469079,
                        0.876953238223366, 0.5467481676210741, 0.1744282580053842, 0.08924552124605833,
                        0.3765565576495673]
    player_ternary = [0.026049308518568903, 0.24955622534606547, 0.22673928889397366, 0.13049151215564336,
                      0.8944196699100078, 0.5526069742201103, 0.42653203231391146, 0.7954012359724127,
                      0.25672816070234417, 0.47182189809712916]

    expected_state = [0, 0, 0, 0.003111401107162237, 0.5021335482597351, 0.85337895154953, 0.09714999794960022,
                      0.7143390774726868, 0.9040235877037048, 0.16265660524368286, 0.5716269612312317,
                      0.2477838397026062, 0.9091309905052185, 0.6592361330986023, 0.7917953133583069,
                      0.27246955037117004, 0.3232679069042206, 0.16762462258338928, 0.8361758589744568,
                      0.8746871948242188, 0.7678519487380981, 0.5267171263694763, 0.5168564319610596,
                      0.8574124574661255, 0.4390338957309723, 0.7161976099014282, 0.9423950910568237,
                      0.4252099096775055, 0.634840726852417, 0.7996163368225098, 0.9607027173042297,
                      0.29866522550582886, 0.030547112226486206, 0.6698791980743408, 0.22825689613819122,
                      0.3147183358669281, 0.1370173990726471, 0.27696337802234927, 0.901607574537501,
                      0.9162636151172863, 0.41581528176420535, 0.3802807190091779, 0.2828253821061679,
                      0.9477133581539083, 0.7817742167252288, 0.06521346143339035, 0.07321247932474795,
                      0.35884491811630737, 0.22579710962370836, 0.7108677439023905, 0.21351596462413647,
                      0.14567335233924572, 0.31729176772958034, 0.011272366424840863, 0.8338832290060092, 0, 0,
                      0.3792401241859009, 0.0664535399729691, 0.9830587561000144, 0.2368589444233229, 0.146776662484974,
                      0.1874666469160684, 0.9514036150101033, 0.4132994067611899, 0.15133628109361186,
                      0.48387899467955564, 0.047234692358380626, 0.2308392882416661, 0.4935608251648853,
                      0.9113517608715533, 0.01738674352128089, 0.7335486816964638, 0.5818931065950903,
                      0.6313772865665637, 0.9449745140714203, 0.02902726416928747, 0.11522208711681747,
                      0.24542619510094865, 0.29808562243890857, 0.6950708345594185, 0.524042067861269,
                      0.48254636506229165, 0, 0, 0, 0, 0, True, True, True, True, 0.19171424900276124]

    state = []
    state.extend(initial)
    state.extend(boost_pad)
    state.extend(ball_state)
    state.extend(player_info)
    state.extend(player_car_state)
    state.extend(player_ternary)

    assert encode_gamestate(GameState(state)) == expected_state