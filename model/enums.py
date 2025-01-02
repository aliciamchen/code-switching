from enum import IntEnum

class AudienceConditions(IntEnum):
    EitherGroup = 0
    OneGroup = 1


class Audiences(IntEnum):
    Ingroup = 1
    Outgroup = 0


class TangramTypes(IntEnum):
    Shared = 0
    GroupSpecific = 1


class Utterances(IntEnum):
    Earlier = 0
    Later = 1

class Conditions(IntEnum):
    ReferEither = 0
    ReferOne = 1
    SocialOne = 2

class Tangram(IntEnum):
    A = 0
    B = 1
    C = 2
    D = 3
    E = 4
    F = 5
    G = 6
    H = 7
    I = 8
    J = 9
    K = 10
    L = 11

class AudienceGroup(IntEnum):
    Red = 0
    Blue = 1

class Counterbalance(IntEnum):
    a = 0
    b = 1