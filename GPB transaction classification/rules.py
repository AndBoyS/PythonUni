import re
import regex
import preprocessing
import pandas as pd
import numpy as np
import pymorphy2
import razdel
import functools

def spaces(x):
    return ' '+x+' '


class RuleStorer:
    rule_dict = {}

    def __init__(self, class_to_code):
        self.class_to_code = class_to_code
    
    def add(self, rule_list):
        for i,_ in enumerate(rule_list[::3]):
            index = rule_list[i*3]
            class_ = rule_list[i*3+1]
            rule = rule_list[i*3+2]
            
            self.rule_dict[index] = {'class':class_,
                            'class code':self.class_to_code[class_],
                            'rule':rule}

    def __getitem__(self, i):
        return self.rule_dict[i]
    
    def __iter__(self):
        return iter(self.rule_dict.keys())


class RulePredictor:

    def __init__(self, all_data, code_names_path, sentences_features_path):

        code_to_class, class_to_code = preprocessing.get_code_names(code_names_path)
        self.morph = pymorphy2.MorphAnalyzer()
        
        self.rules = RuleStorer(class_to_code)

        self.rules.add([
            0, 'аренда', lambda x: regex.search(r'(?<!возврат.*)(?<!финанс.*)(?<!лизинг.*)аренда(?!.*лизинг)(?!.*финанс)', x['purp']),
            1, 'возврат', lambda x: 'недостача' in x['purp'],
            2, 'выручка', lambda x: 'цессия' in x['purp'] and 'концессия' not in x['purp'] and x['name_to'] == 'Коммерческие организации',
            3, 'выплаты соцхарактера', lambda x: regex.search(r'(?<!алименты.*)дотация(?!.*алименты)', x['purp']) and x['name_to'] == 'Физические лица',
            4, 'возврат', lambda x: regex.search(r'(?<!депозит.*?)(?<!заём.*?)возврат(?!.*заём)(?!.*депозит)', x['purp']) and 'безвозвратный' not in x['purp'],
            5, 'страховое возмещение', lambda x: 'страховой выплата' in x['purp'],
            6, 'страховая премия', lambda x: 'не бояться перемена' in x['purp'],
            7, 'страховая премия',lambda x: regex.search(r'(?<!возврат.*?)страховой премия', x['purp']),
            8, 'зарплата', lambda x: regex.search(r'(?<!командиров.*)(?<!аренда.*)(?<!возврат.*)аванс(?!.*аренда)(?!.*отчёт)(?!.*по трудовой договор)(?!.*командиров)(?!.*поездка)', x['purp']) and x['name_to'] == 'Физические лица',
            9, 'эквайринг', lambda x: 'операция покупка' in x['purp'],
            10, 'выручка', lambda x: regex.search(r'(?<!доплата.*)(?<!пеня.*)(?<!заём.*)(?<!аренда.*)(?<!инкас.*)оплата(?!.*аренда)(?!.*инкас)(?!.*заём)(?!.*пеня)(?!.*доплата)', x['purp']) and (x['name_to'] == 'Коммерческие организации' or x['name_to'] == 'Индивидуальные предприниматели'),
            11, 'лизинг', lambda x: regex.search(r'(?<!газпромбанк.*)лизинг(?!.*газпромбанк)', x['purp']),
            13, 'выплаты соцхарактера', lambda x: 'помощь' in x['purp'] and x['name_to'] != 'Касса кредитных организаций',
            14, 'дивиденды', lambda x: 'пао газпром нефть за 9 месяц' in x['purp'],
            15, 'прочее', lambda x: 'по заявка на погашение инвестиционный пай' in x['purp'] or 'пай опиф' in x['purp'],
            16, 'налоги прочие', lambda x: ('госпошлина' in x['purp'] or 'государственный пошлина' in x['purp']) and x['name_to'] == 'Доходы, распределяемые органами Федерального казначейства между бюджетами бюджетной системы Российской Федерации',
            17, 'перевод на фл', lambda x: regex.search(r'(?<!возврат.*)подотчёт', x['purp']) or regex.search(r'(?<!возврат.*)под отчёт', x['purp']),
            18, 'комиссии банку', lambda x: 'комиссия' in x['purp'] and x['name_to'] in ['Требования по прочим операциям', 'Доходы'],
            19, 'перевод на фл', lambda x: 'расход' in x['purp'] and x['name_to'] == 'Физические лица',
            21, 'выручка', lambda x: 'юрист 24' in x['purp'],
            22, 'страховая премия', lambda x: 'тот самый случай' in x['purp'],
            23, 'выплаты соцхарактера', lambda x: 'по уплата половина процент по ипотечный кредит' in x['purp'] or 'по уплата половина первоначальный взнос по ипотечный кредит' in x['purp'],
            24, 'налоги ндфл', lambda x: regex.search(r'(?<!штраф.*?)(?<!пеня.*?)(?<!пенить.*?)налог на доход(?!.*удержать)', x['purp']),
            25, 'выручка', lambda x: 'на ведение иис' in x['purp'],
            26, 'выплаты соцхарактера', lambda x: 'всх' in x['purp'] and x['name_to'] != 'Касса кредитных организаций',
            27, 'прочее', lambda x: regex.search(r'заявление о закрытие по договор', x['purp']),
            #28, 'пополнение счёта', lambda x: 'заявление о закрытие счёт' in x['purp'],
            29, 'выручка', lambda x: regex.search(r'(?<!возврат.*)оплата за а м', x['purp']) or regex.search(r'(?<!возврат.*)оплата за автомобиль', x['purp']),
            30, 'возврат', lambda x: 'списание ошибочно зачислить' in x['purp'],
            31, 'кредит', lambda x: regex.search(r'(?<!заём.*?)(?<!процент.*?)(?<!возврат.*)транш(?!.*заём)(?!.*процент)', x['purp']),
            32, 'налоги ндфл', lambda x: 'оплата ндфл' in x['purp'] and x['name_to'] == 'Доходы, распределяемые органами Федерального казначейства между бюджетами бюджетной системы Российской Федерации',
            33, 'валюта', lambda x: 'конверсия' in x['purp'] and x['name_to'] == 'Расчеты по конверсионным операциям, производным финансовым инструментам и прочим договорам (сделкам), по которым расчеты и поставка осуществляются не ранее следующего дня после дня заключения договора (сделки)',
            34, 'оплата фл', lambda x: 'подряд' in x['purp'] and x['name_to'] == 'Физические лица',
            35, 'выручка', lambda x: regex.search(r'(?<!возврат.*)(?<!штраф.*)подряд(?!штраф.*)', x['purp']) and x['name_to'] == 'Коммерческие организации',
            36, 'штрафы государство', lambda x: regex.search(r'штраф.*гибдд', x['purp']),
            37, 'штрафы государство', lambda x: 'административный штраф' in x['purp'] or 'штраф административный' in x['purp'],
            #38, 'пополнение счёта', lambda x: 'для покупка ценный бумага' in x['purp'] or 'для покупка цена бумага' in x['purp'],
            39, 'пожертвования и благотворительность', lambda x: 'благотворительный взнос' in x['purp'],
            40, 'страховое возмещение', lambda x: 'выплата страхователь' in x['purp'],
            41, 'страховое возмещение', lambda x: 'страховой возмещение' in x['purp'],
            42, 'дивиденды', lambda x: regex.search(r'(?<!ндфл.*)(?<!кредит.*)дивиденд(?!.*кредит)(?!.*плата ндфл)', x['purp']),
            43, 'вексель', lambda x: 'вексель' in x['purp'],
            44, 'инкассирование', lambda x: 'оплата за инкассация' in x['purp'],
            45, 'налоги ндс', lambda x: 'оплата ндс' in x['purp'] and 'доплата' not in x['purp'],
            46, 'налоги прибыль', lambda x: regex.search(r'(?<!штраф.*)(?<!возврат.*)(?<!пеня.*?)(?<!пенить.*?)налог на прибыль(?!.*штраф)', x['purp']) and x['name_to'] == 'Доходы, распределяемые органами Федерального казначейства между бюджетами бюджетной системы Российской Федерации',
            47, 'налоги прочие', lambda x: regex.search(r'(?<!возмещение.*)(?<!ндфл с.*)(?<!пеня.*?)(?<!пенить.*?)земельный налог', x['purp']) and x['name_to'] == 'Доходы, распределяемые органами Федерального казначейства между бюджетами бюджетной системы Российской Федерации',
            48, 'налоги прочие', lambda x: regex.search(r'(?<!возмещение.*)(?<!ндфл с.*)(?<!пеня.*?)(?<!пенить.*?)водный налог', x['purp']) and x['name_to'] == 'Доходы, распределяемые органами Федерального казначейства между бюджетами бюджетной системы Российской Федерации',
            49, 'налоги прочие', lambda x: regex.search(r'(?<!возмещение.*)(?<!ндфл с.*)(?<!пеня.*?)(?<!пенить.*?)транспортный налог', x['purp']) and x['name_to'] == 'Доходы, распределяемые органами Федерального казначейства между бюджетами бюджетной системы Российской Федерации',
            50, 'зарплата', lambda x: regex.search(r'алименты', x['purp']) and x['name_from'] == 'Коммерческие организации',
            51, 'зарплата', lambda x: 'пенсионный взнос' in x['purp'] and x['name_to'] == 'Финансовые организации' and x['name_from'] == 'Коммерческие организации',
            52, 'выручка', lambda x: (regex.search(r'(?<!льгота.*)(?<!возврат.*)(?<!пеня.*)(?<!пенить.*)(?<!аренда.*)электроэнергия(?!.*аренда)(?!.*льгота)', x['purp']) or regex.search(r'(?<!льгота.*)(?<!возврат.*)(?<!пеня.*)(?<!пенить.*)(?<!аренда.*)эл энергия(?!.*аренда)(?!.*льгота)', x['purp']) or regex.search(r'(?<!льгота.*)(?<!возврат.*)(?<!пеня.*)(?<!пенить.*)(?<!аренда.*)эл эн(?!.*аренда)(?!.*льгота)', x['purp']) or regex.search(r'(?<!льгота.*)(?<!возврат.*)(?<!пеня.*)(?<!пенить.*)(?<!аренда.*)энергоснабжение(?!.*аренда)(?!.*льгота)', x['purp'])) and x['name_to'] in ['Коммерческие организации', 'Счет для идентификации платежа', 'Индивидуальные предприниматели'],
            53, 'проценты по кредиту', lambda x: regex.search(r'списание задолж.*по процент', x['purp']),
            54, 'кредит', lambda x: regex.search(r'списание задолж.*по осн долг', x['purp']),
            55, 'пожертвования и благотворительность', lambda x: 'пожертвование' in x['purp'],
            56, 'выручка', lambda x: regex.search(r'выплата процент.*по сделка', x['purp']),
            57, 'выручка', lambda x: 'соцнавигатор' in x['purp'],
            58, 'выручка', lambda x: 'от покупатель кпп' in x['purp'],
            59, 'выручка', lambda x: regex.search(r'(?<!возврат.*)(?<!лизинг.*)(?<!аренда.*)аванс(?!.*лизинг)(?!.*аренда)', x['purp']) and x['name_to'] in ['Коммерческие организации', 'Индивидуальные предприниматели'],
            60, 'перевод на фл', lambda x: regex.search(r'(?<!возврат.*)авансовый отчёт', x['purp']),
            61, 'зарплата', lambda x: regex.search(r'(?<!возврат.*)(?<!страх.*)(?<!проф.*)(?<!член.*)(?<!праздн.*)премия(?!.*проф)(?!.*праздн)(?!.*член)', x['purp']) and x['name_to'] == 'Физические лица',
            62, 'зарплата', lambda x: 'больничный' in x['purp'],
            63, 'зарплата', lambda x: 'оплата труд' in x['purp'] and x['name_to'] == 'Физические лица',
            64, 'зарплата', lambda x: regex.search(r'(?<!возмещ.*)(?<!под.*отчёт.*)(?<!аванс.*)(?<!мп.*)(?<!компенсация.*)(?<!проезд.*)(?<!соц.*)(?<!помощь.*)отпуск(?!.*помощь)(?!.*соц)(?!.*мп)(?!.*компенсация)(?!.*аванс)(?!.*возмещ)(?!.*под.*отчёт)', x['purp']) and x['name_to'] == 'Физические лица',
            65, 'зарплата', lambda x: 'при увольнение' in x['purp'],
            66, 'выручка', lambda x: 'окончательный расчёт' in x['purp'] and x['name_to'] == 'Коммерческие организации',
            67, 'зарплата', lambda x: 'окончательный расчёт' in x['purp'] and x['name_to'] == 'Физические лица',
            68, 'займ', lambda x: regex.search(r'(?<!аванс.*)(?<!помощь.*)(?<!ценный бумага.*)заём(?!.*ценный бумага)(?!.*помощь)', x['purp']),
            69, 'перевод на фл', lambda x: regex.search(r'(?<!возврат.*)командировочный', x['purp']),
            70, 'кредит', lambda x: 'согласно кредит' in x['purp'],
            #71, 'медосмотр', lambda x: 'медосмотр' in x['purp'] and x['name_to'] == 'Физические лица',
            #72, 'удержание', lambda x: 'удержание' in x['purp'] and x['name_to'] == 'Физические лица',
            73, 'выручка', lambda x: regex.search(r'отчисление от.*фзп', x['purp']),
            74, 'зарплата', lambda x: regex.search(r'(?<!ндфл.*)(?<!пеня.*)(?<!мат.*)(?<!доплата.*)нетруд', x['purp']),
            75, 'выплаты соцхарактера', lambda x: regex.search(r'из чл.*взнос', x['purp']),
            76, 'страховая премия', lambda x: regex.search(r'страховой взнос(?!.*пенс)', x['purp']),
            77, 'штрафы прочие', lambda x: 'неустойка' in x['purp'],
            78, 'проценты по кредиту', lambda x: regex.search(r'процент.*по кредитный договор', x['purp']) or regex.search(r'процент.*по кредитный соглашение', x['purp']),
            79, 'выплаты соцхарактера', lambda x: regex.search(r'возмещ.*процент.*по ипотеч.*кредит', x['purp']),
            80, 'кредит', lambda x: 'в цель исполнение обязательство по кредитный договор' in x['purp'],
            81, 'выручка', lambda x: 'нкд' in x['purp'],
            82, 'взыскание с фл', lambda x: regex.search(r'(?<!штраф.*)(?<!суд.*)взыскание(?!.*суд)(?!.*штраф)(?!.*удостоверение)', x['purp']),
            83, 'выручка', lambda x: regex.search(r'(?<!возврат.*)(?<!заём.*)(?<!а м.*)(?<!инкас.*)комиссия(?!.*заём)(?!.*а м)(?!.*инкас)', x['purp']) and x['name_to'] == 'Коммерческие организации',
            84, 'страховая премия', lambda x: 'пенсионный взнос в польза' in x['purp'] and x['name_from'] == 'Физические лица',
            85, 'перевод на фл', lambda x: ('собственн' in x['purp'] or 'личн' in x['purp']) and x['name_from'] == 'Индивидуальные предприниматели' and x['name_to'] in ['Физические лица', 'Депозиты до востребования', 'Депозиты на срок от 1 года до 3 лет', 'Депозиты на срок от 181 дня до 1 года', 'Депозиты на срок от 91 до 180 дней'],
            86, 'зарплата', lambda x: regex.search(r'проф.*взнос', x['purp']) and x['name_to'] == 'Некоммерческие организации' and x['name_from'] == 'Коммерческие организации',
            87, 'зарплата', lambda x: (regex.search(r'отчисление.*фзп', x['purp']) or regex.search(r'отчисление.*фот', x['purp'])) and x['name_from'] == 'Коммерческие организации',
            88, 'зарплата', lambda x: 'негос' in x['purp'] and x['name_from'] == 'Коммерческие организации',
            89, 'выплаты соцхарактера', lambda x: 'компенсация затрата член профсоюз' in x['purp'],
            90, 'зарплата', lambda x: 'ппо' in x['purp'] and x['name_from'] == 'Коммерческие организации' and x['name_to'] == 'Некоммерческие организации',
            91, 'выплаты соцхарактера', lambda x: regex.search(r'(?<!возврат.*)путёвка', x['purp']) and x['name_to'] == 'Физические лица',
            92, 'выплаты соцхарактера', lambda x: regex.search(r'поощрение.*членский', x['purp']),
            94, 'взыскание с фл', lambda x: (regex.search(r'удержан.*с', x['purp']) or 'взыскание с' in x['purp']) and x['name_from'] == 'Коммерческие организации',
            95, 'налоги ндс', lambda x: 'доплата ндс' in x['purp'],
            96, 'штрафы государство', lambda x: ('штраф по налог' in x['purp'] or 'пеня по налог' in x['purp']) and x['name_to'] == 'Доходы, распределяемые органами Федерального казначейства между бюджетами бюджетной системы Российской Федерации',
            97, 'штрафы прочие', lambda x: 'по трудовой спор' in x['purp'],
            98, 'налоги прибыль',  lambda x: ('перечислить для уплата налог на прибыль' in x['purp'] or 'авансовый платёж по налог на прибыль' in x['purp']),
            99, 'суды', lambda x: 'возмещение судебный расход' in x['purp'],
            100, 'суды', lambda x: ('списание по исполнительный документ' in x['purp'] or 'взыскание расход за производство судебный экспертиза' in x['purp']),
            101, 'суды', lambda x: regex.search(r'(?<!труд.*)взыскание согласно(?!.*труд)', x['purp']),
            102, 'выручка', lambda x: 'гашение кредиторский задолженность' in x['purp'],
            103, 'выручка', lambda x: 'нкд' in x['purp'],
            104, 'выручка', lambda x: ('допуск к клиринг обязательство' in x['purp'] or 'задолженность за рко' in x['purp']),
            105, 'обеспечение', lambda x: regex.search(r'перечисление денежный средств для обеспечение.*участие в закупочный процедура', x['purp']),
            106, 'комиссии банку', lambda x: 'операция снятие наличный' in x['purp'],
            107, 'комиссии банку', lambda x: ('открытие счёт' in x['purp'] or 'открытие р с' in x['purp'] or 'начисление процент' in x['purp']) and x['name_to'] == 'Доходы',
            108, 'прочее', lambda x: regex.search(r'выплата.*устав', x['purp']),
            109, 'оплата фл', lambda x: 'авторский вознаграждение' in x['purp'],
            111, 'суды', lambda x: 'арбитражный сбор для подача иск' in x['purp'],
            112, 'оплата фл', lambda x: 'оплата подработка' in x['purp'],
            113, 'налоги прочие', lambda x: 'сбор за продление срок действие лицензия' in x['purp'],
            114, 'докапитализация', lambda x: 'дополнительный взнос в уставный капитал' in x['purp'],
            115, 'штрафы прочие', lambda x: 'сальдо' in x['purp'] and x['name_to'] == 'Коммерческие организации',
            116, 'штрафы прочие', lambda x: 'пеня' in x['purp'] and x['name_to'] == 'Коммерческие организации',
            117, 'штрафы прочие', lambda x: 'штраф по просрочка' in x['purp'] and x['name_to'] == 'Коммерческие организации',
            118, 'штрафы государство', lambda x: 'пеня' in x['purp'] and x['name_to'] == 'Доходы, распределяемые органами Федерального казначейства между бюджетами бюджетной системы Российской Федерации',
            119, 'прочее', lambda x: 'дилерский услуга' in x['purp'],
            120, 'прочее', lambda x: 'выписка с счёт депо раздел счёт депо' in x['purp'],
            121, 'прочее', lambda x: 'иной выписка справка для депонент' in x['purp'],
            122, 'обеспечение', lambda x: 'обеспечительный платёж по договор аренда' in x['purp'],
            123, 'кредит', lambda x: 'частичный возврат основной долг по кредитный соглашение' in x['purp'],
            124, 'зарплата', lambda x: 'нпо' in x['purp'] and x['name_from'] == 'Коммерческие организации',
            125, 'страховая премия', lambda x: 'нпо' in x['purp'] and x['name_from'] in ['Финансовые организации', 'Физические лица'],
            127, 'выручка', lambda x: regex.search(r'оплата по приложение.*к дилерский договор', x['purp']),
            128, 'налоги ндс', lambda x: 'ндс с комиссия' in x['purp'] and x['name_from'] == 'Требования по прочим операциям' and x['name_to'] == 'Налог на добавленную стоимость, полученный',
            129, 'налоги прочие', lambda x: 'в свп' in x['purp'] and x['name_from'] == 'Коммерческие организации' and x['name_to'] == 'Коммерческие организации',
            130, 'налоги прочие', lambda x: ('пополнение рз' in x['purp'] or 'пополнение расчётный запись' in x['purp'])  and x['name_from'] == 'Коммерческие организации' and x['name_to'] == 'Коммерческие организации',
            131, 'докапитализация', lambda x: 'взнос в уставный' in x['purp'],
            132, 'проценты по кредиту', lambda x: 'требование по получение процент по договор' in x['purp'] and x['name_from'] == 'Начисленные проценты по предоставленным (размещенным) денежным средствам' and x['name_to'] == 'Доходы',
            133, 'проценты по кредиту', lambda x: 'погашение процент' in x['purp'] and x['name_to'] in ['Начисленные проценты по предоставленным (размещенным) денежным средствам', 'Коммерческие организации', 'Обязательства по прочим операциям'] and x['name_from'] in ['Коммерческие организации', 'Отдельный счет головного исполнителя, исполнителя государственного оборонного заказа'],
            134, 'проценты по кредиту', lambda x: regex.search(r'списание задолж.*согласно.*по процент', x['purp']) and x['name_to'] == 'Начисленные проценты по предоставленным (размещенным) денежным средствам' and x['name_from'] == 'Коммерческие организации',
            135, 'кредит', lambda x: regex.search(r'списание задолж.*согласно.*по осн.*долг', x['purp']),
            136, 'зарплата', lambda x: ('страховой взнос' in x['purp'] or 'взнос на страховой' in x['purp']) and x['name_from'] == 'Индивидуальные предприниматели' and x['name_to'] == 'Доходы, распределяемые органами Федерального казначейства между бюджетами бюджетной системы Российской Федерации',
            137, 'кредит', lambda x: regex.search(r'кредит.*задолж', x['purp']) and x['name_from'] == 'Коммерческие организации' and x['name_to'] == 'Коммерческие организации',
            138, 'выручка', lambda x: regex.search(r'(?<!аренда.*)оплата.*задолж(?!.*аренда)', x['purp']) and x['name_from'] == 'Коммерческие организации' and x['name_to'] == 'Коммерческие организации',
            139, 'прочее', lambda x: 'выплата выкупной сумма наследник' in x['purp'] and x['name_from'] == 'Финансовые организации' and x['name_to'] == 'Физические лица',
            140, 'прочее', lambda x: 'передача средство пенсионный резерв в ду' in x['purp'] and x['name_from'] == 'Коммерческие организации' and x['name_to'] == 'Финансовые организации',
            141, 'прочее', lambda x: regex.search(r'(?<!возврат.*)уступка', x['purp']) and x['name_from'] == 'Коммерческие организации' and x['name_to'] == 'Физические лица',
            142, 'прочее', lambda x: x['name_from'] == 'Резервы на возможные потери',
            143, 'возврат', lambda x: x['name_from'] == 'Физические лица' and x['name_to'] == 'Расчеты с работниками по подотчетным суммам',
            144, 'налоги ндс', lambda x: 'доплата 2' in x['purp'],
            146, 'дивиденды', lambda x: 'учредитель' in x['purp'],
            147, 'страховая премия', lambda x: ('осаго' in x['purp'] or 'каско' in x['purp']) and x['name_from'] == 'Коммерческие организации' and (x['name_to'] == 'Коммерческие организации' or x['name_to'] == 'Финансовые организации'),
            148, 'обеспечение', lambda x: 'возврат обеспеч' in x['purp'] and x['name_from'] == 'Коммерческие организации' and x['name_to'] == 'Коммерческие организации',
            149, 'штрафы прочие', lambda x: 'неустой' in x['purp'] and x['name_from'] == 'Коммерческие организации' and x['name_to'] == 'Коммерческие организации',
            150, 'прочее', lambda x: 'репо' in x['purp'],])
        
        #шаблоны
        #regex.search(r'', x['purp'])
        #'' in x['purp']

        #собираем удаленные классы в отрицательный класс
        for i in self.rules:
            class_ = self.rules[i]['class code']
            if class_ not in all_data['class'].unique():
                self.rules[i]['class'] = '__прочее'
                self.rules[i]['class code'] = -1

        #{словосочетание : номер класса}
        sentences_features = pd.read_excel(sentences_features_path)
        self.sentence_to_class = {}

        for i,sentences in sentences_features['Однозначно'].items():

            cls = sentences_features['Классы'][i].lower()

            if type(sentences) == float:
                continue

            if cls in class_to_code.index:
                class_number = class_to_code[cls]
                for sentence in tuple(sentences.lower().split(';')):
                    if sentence != '':
                        self.sentence_to_class[sentence.strip()] = class_number

        #собираем удаленные классы в отрицательный класс
        for sentence in self.sentence_to_class:
            if self.sentence_to_class[sentence] not in all_data['class'].unique():
                self.sentence_to_class[sentence] = -1

    @functools.lru_cache(maxsize=5000)
    def lemmatize(self, word):
        return self.morph.parse(word)[0].normalized.word

    def preprocess_for_conds(self, text):
        '''
        Предобработка строки для правил
        '''

        text = text.lower()
        tokens = list(razdel.tokenize(text))

        return ' '.join([self.lemmatize(t.text) for t in tokens if regex.search(r'^[\w\d]*$', t.text)])

    def _classify_one_with_conds(self, row, collect_cond_stats=True): #collect_cond_stats нужно для сбора статистики о правилах и уменьшает скорость программы

        row.loc['purp'] = self.preprocess_for_conds(row['purp'])
        prediction = None

        for i in self.rules:

            cls = self.rules[i]['class code']
            cond = self.rules[i]['rule']

            cond_state = cond(row)

            if cond_state:

                if collect_cond_stats:

                    self.conds_summary[i][1] += 1
                    if cls == row['class']:
                        self.conds_summary[i][0] += 1
            
                if prediction is None:
                    prediction = cls
                
                elif prediction != cls:
                    prediction = 'duplicate'
                    
                    if not collect_cond_stats:
                        break


        prediction = None if prediction == 'duplicate' else prediction
            
        return prediction

    def _classify_one_with_sentences(self, sentence):

        prediction = None

        for key_sentence in self.sentence_to_class:

            cond_state = spaces(key_sentence) in spaces(sentence)
            
            if cond_state and prediction is None:
                prediction = self.sentence_to_class[key_sentence]
            
            elif cond_state and prediction != self.sentence_to_class[key_sentence]:
                prediction = 'duplicate'
                break

        prediction = None if prediction == 'duplicate' else prediction

        return prediction

    def classify_with_rules(self, data, use_conds=True, use_key_sentences=True): 
        '''
        Классификация при помощи правил и ключевых слов

        Parameters:
        data: pd.DataFrame - данные для классификации
        use_conds: bool - использовать ли правила на оснвое регулярных выражений
        use_key_sentences: bool - использовать ли ключевые словосочетания

        Returns:
        all_preds: list - список предсказанных классов (если предсказания для строки нет, в списке будет значение None)
        '''

        data = data.copy()
        all_preds = []
        self.conds_summary = {i:[0,0] for i in self.rules} #статистика правильных предсказаний для каждого правила в виде "сколько правильно предсказано":"сколько всего предсказано"

        for _,row in data.iterrows():

            row['purp'] = row['purp'].lower()

            prediction = self._classify_one_with_conds(row) if use_conds else None
            prediction = self._classify_one_with_sentences(row['purp']) if prediction is None and use_key_sentences else prediction

            all_preds.append(prediction)

        return all_preds


    def _get_one_hot_features_one_row(self, row):

        row_preprocessed = row.copy() #предобработанная строка для правил
        row_preprocessed.loc['purp'] = self.preprocess_for_conds(row['purp'])

        classes_dict = {}

        for i in self.rules:
        
            cls = self.rules[i]['class code']
            cond = self.rules[i]['rule']

            classes_dict[cls] = classes_dict.get(cls, 0) + bool(cond(row_preprocessed))

        for key_sentence in self.sentence_to_class:

            cls = self.sentence_to_class[key_sentence]

            classes_dict[cls] = classes_dict.get(cls, 0) + (spaces(key_sentence) in spaces(row['purp']))

        tuple_list = sorted(classes_dict.items(), key=lambda x: x[0]) #сортируем по идентификатору классов

        return [val for cls,val in tuple_list]

    def get_one_hot_features(self, data):
        '''
        Получить признаки на основе правил и ключевых слов в виде one-hot векторов (каждый столбец отвечает за один из классов)

        Parameters:
        data: pd.DataFrame - данные

        Returns:
        features: np.array
        '''

        data = data.copy()
        features = []

        for i,row in data.iterrows():

            row['purp'] = row['purp'].lower()

            features.append(self._get_one_hot_features_one_row(row))

        return np.array(features)
