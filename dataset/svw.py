from typing import List

from dataset.abc_dataset import AbcDataset, Label


class SVW(AbcDataset):
    BASKETBALL = Label('basketball')
    FOOTBALL = Label('football')
    GYMNASTICS = Label('gymnastics')
    BOXING = Label('boxing')
    TENNIS = Label('tennis')
    VOLLEYBALL = Label('volleyball')
    ARCHERY = Label('archery')
    GOLF = Label('golf')
    HURDLING = Label('hurdling')
    RUNNING = Label('running')
    BASEBALL = Label('baseball')
    CHEERLEADING = Label('cheerleading')
    JAVELIN = Label('javelin')
    SHOTPUT = Label('shotput')
    DISCUSTHROW = Label('discusthrow')
    HAMMERTHROW = Label('hammerthrow')
    LONGJUMP = Label('longjump')
    SKATING = Label('skating')
    BMX = Label('bmx')
    DIVING = Label('diving')
    HIGHJUMP = Label('highjump')
    POLEVAULT = Label('polevault')
    SKIING = Label('skiing')
    WEIGHT = Label('weight')
    BOWLING = Label('bowling')
    HOCKEY = Label('hockey')
    ROWING = Label('rowing')
    SOCCER = Label('soccer')
    WRESTLING = Label('wrestling')
    SWIMMING = Label('swimming')

    @classmethod
    def get_all_classes(cls) -> List[Label]:
        return [
            cls.BASKETBALL, cls.FOOTBALL, cls.GYMNASTICS, cls.BOXING, cls.TENNIS, cls.VOLLEYBALL, cls.ARCHERY, cls.GOLF,
            cls.HURDLING, cls.RUNNING, cls.BASEBALL, cls.CHEERLEADING, cls.JAVELIN, cls.SHOTPUT,
            cls.DISCUSTHROW, cls.HAMMERTHROW, cls.LONGJUMP, cls.SKATING, cls.BMX, cls.DIVING, cls.HIGHJUMP,
            cls.POLEVAULT, cls.SKIING, cls.WEIGHT, cls.BOWLING, cls.HOCKEY, cls.ROWING, cls.SOCCER, cls.WRESTLING]
