from enum import Enum

from rlgym_sim.utils.gamestates import GameState


def encode_gamestate(state: GameState):
    state_vals = [0, state.blue_score, state.orange_score]
    state_vals += state.boost_pads.tolist()

    for bd in (state.ball, state.inverted_ball):
        state_vals += bd.position.tolist()
        state_vals += bd.linear_velocity.tolist()
        state_vals += bd.angular_velocity.tolist()

    for p in state.players:
        state_vals += [p.car_id, p.team_num]
        for cd in (p.car_data, p.inverted_car_data):
            state_vals += cd.position.tolist()
            state_vals += cd.quaternion.tolist()
            state_vals += cd.linear_velocity.tolist()
            state_vals += cd.angular_velocity.tolist()
        state_vals += [
            p.match_goals,
            p.match_saves,
            p.match_shots,
            p.match_demolishes,
            p.boost_pickups,
            p.is_demoed,
            p.on_ground,
            p.ball_touched,
            p.has_jump,
            p.has_flip,
            p.boost_amount
        ]
    return state_vals


# Now some constants for easy and consistent querying of gamestate values
class StateConstants:
    DUMMY = 0
    BLUE_SCORE = 1
    ORANGE_SCORE = 2
    BOOST_PADS = slice(3, 3 + GameState.BOOST_PADS_LENGTH)
    BALL_POSITION = slice(BOOST_PADS.stop, BOOST_PADS.stop + 3)
    BALL_LINEAR_VELOCITY = slice(BALL_POSITION.stop, BALL_POSITION.stop + 3)
    BALL_ANGULAR_VELOCITY = slice(BALL_LINEAR_VELOCITY.stop, BALL_LINEAR_VELOCITY.stop + 3)

    PLAYERS = slice(BALL_ANGULAR_VELOCITY.stop + 9, None)  # Skip inverted data
    CAR_IDS = slice(0, None, GameState.PLAYER_INFO_LENGTH)
    TEAM_NUMS = slice(CAR_IDS.start + 1, None, GameState.PLAYER_INFO_LENGTH)
    CAR_POS_X = slice(TEAM_NUMS.start + 1, None, GameState.PLAYER_INFO_LENGTH)
    CAR_POS_Y = slice(CAR_POS_X.start + 1, None, GameState.PLAYER_INFO_LENGTH)
    CAR_POS_Z = slice(CAR_POS_Y.start + 1, None, GameState.PLAYER_INFO_LENGTH)
    CAR_QUAT_W = slice(CAR_POS_Z.start + 1, None, GameState.PLAYER_INFO_LENGTH)
    CAR_QUAT_X = slice(CAR_QUAT_W.start + 1, None, GameState.PLAYER_INFO_LENGTH)
    CAR_QUAT_Y = slice(CAR_QUAT_X.start + 1, None, GameState.PLAYER_INFO_LENGTH)
    CAR_QUAT_Z = slice(CAR_QUAT_Y.start + 1, None, GameState.PLAYER_INFO_LENGTH)
    CAR_LINEAR_VEL_X = slice(CAR_QUAT_Z.start + 1, None, GameState.PLAYER_INFO_LENGTH)
    CAR_LINEAR_VEL_Y = slice(CAR_LINEAR_VEL_X.start + 1, None, GameState.PLAYER_INFO_LENGTH)
    CAR_LINEAR_VEL_Z = slice(CAR_LINEAR_VEL_Y.start + 1, None, GameState.PLAYER_INFO_LENGTH)
    CAR_ANGULAR_VEL_X = slice(CAR_LINEAR_VEL_Z.start + 1, None, GameState.PLAYER_INFO_LENGTH)
    CAR_ANGULAR_VEL_Y = slice(CAR_ANGULAR_VEL_X.start + 1, None, GameState.PLAYER_INFO_LENGTH)
    CAR_ANGULAR_VEL_Z = slice(CAR_ANGULAR_VEL_Y.start + 1, None, GameState.PLAYER_INFO_LENGTH)
    MATCH_GOALS = slice(CAR_ANGULAR_VEL_Z.start + 1 + 13, None, GameState.PLAYER_INFO_LENGTH)  # Skip inverted data
    MATCH_SAVES = slice(MATCH_GOALS.start + 1, None, GameState.PLAYER_INFO_LENGTH)
    MATCH_SHOTS = slice(MATCH_SAVES.start + 1, None, GameState.PLAYER_INFO_LENGTH)
    MATCH_DEMOLISHES = slice(MATCH_SHOTS.start + 1, None, GameState.PLAYER_INFO_LENGTH)
    BOOST_PICKUPS = slice(MATCH_DEMOLISHES.start + 1, None, GameState.PLAYER_INFO_LENGTH)
    IS_DEMOED = slice(BOOST_PICKUPS.start + 1, None, GameState.PLAYER_INFO_LENGTH)
    ON_GROUND = slice(IS_DEMOED.start + 1, None, GameState.PLAYER_INFO_LENGTH)
    BALL_TOUCHED = slice(ON_GROUND.start + 1, None, GameState.PLAYER_INFO_LENGTH)
    HAS_JUMP = slice(BALL_TOUCHED.start + 1, None, GameState.PLAYER_INFO_LENGTH)
    HAS_FLIP = slice(HAS_JUMP.start + 1, None, GameState.PLAYER_INFO_LENGTH)
    BOOST_AMOUNT = slice(HAS_FLIP.start + 1, None, GameState.PLAYER_INFO_LENGTH)
