# House Prices: Advanced Regression Techniques

## Инструкция по запуску

```
cd statistics
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

## Описание приложения

[House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) — это датасет на платформе Kaggle, предназначенный для задач регрессии. Основная цель заключается в предсказании итоговой цены на жильё. Датасет содержит различные аспекты информации о жилой недвижимости из города Ames, штат Айова, США.

Датасет включает 79 переменных, описывающих характеристики жилых домов в Ames, Iowa. Ниже представлены некоторые из них:

1. MSSubClass: Класс здания, влияющий на продажу.
2. LotFrontage: Длина улицы, примыкающей к участку.
3. LotArea: Размер участка в квадратных футах.
4. LandSlope: Наклон участка.
5. Condition1: Различные условия, находящиеся неподалёку (например, близость к основной дороге).
6. BldgType: Тип жилья.
7. HouseStyle: Стиль жилища.
8. OverallQual: Общая оценка материалов и отделки дома.
9. OverallCond: Общее состояние дома.
10. YearBuilt: Год постройки.
11. RoofStyle и RoofMatl: Стиль и материал крыши.
12. Exterior1st и Exterior2nd: Внешняя отделка дома.
13. MasVnrType и MasVnrArea: Тип и площадь кладки облицовочного камня.
14. ExterQual и ExterCond: Качество и состояние внешних материалов.

### Целевая переменная:
SalePrice

### Исследовательский вопрос:
Какие характеристики недвижимости могут служить предикторами ее стоимости?

## Аналитика данных в приложении

В приложении есть 3 основные вкладки, позволяющие провести анализ данного датасета.
Опишем процесс аналитики с помощью данного приложения.

### 1. Exploratory Data Analysis (EDA)
здесь описываются основные статистики, а также графики распределения для признаков и целевой переменной.
Примеры анализа некоторых признаков:

SalePrice (цена продажи):
   - Распределение: ближе нормальному, но со смещением вправо (положительная асимметрия).
   - Обработка: Логарифмирование этого признака может помочь нормализовать распределение, что улучшит результаты линейных регрессионных моделей.

LotArea (размер участка в квадратных футах):
   - Распределение: Имеет сильно положительное смещение; большинство участков малы по размеру с некоторыми крайне большими выбросами.
   - Обработка: Применение логарифмирования или иных преобразований, например, корень квадратный, может помочь уменьшить эффект выбросов и сделать распределение более симметричным.

YearBuilt (год постройки):
   - Распределение: Не является нормальным; возможно, модальное с большим количеством новых домов, и меньшим — старых.
   - Обработка: Может использоваться как категориальный признак или преобразовываться в непрерывный признак, который показывает возраст дома относительно года продажи.

OverallQual (общая оценка качества материалов и отделки дома):
   - Распределение: Категориальное оценочное, больше близко к нормальному с медианой 6.

CentralAir (наличие системы кондиционирования):
   - Распределение: Двоичный признак (Y/N). Скорее всего, в 93% случаев присутствует.
   - Обработка: Можно закодировать как 0 и 1, где 1 - наличие кондиционирования.

GarageCars (размер гаража по вместимости автомобилей):
   - Распределение: многие дома имеют гаражи с вместимостью 2 машин, а вместимость для 4 машин встречается очень редко.

### 2. Correlation and Cluster Analysis: корреляционный и кластерный анализ

#### Посмотрим на корреляционную матрицу по всем признакам. Наибольшую корреляцию ожидаемо имеют следующие признаки:
GarageCars - GarageArea: 0.88 (Оба этих признака имеют сильную корреляцию с ценой продажи, поскольку они отражают вместимость и, соответственно, удобство гаража)
YearBuilt - GarageYrBlt: 0.83
GrLivArea - TotRmsAbvGrd: 0.83
TotalBsmtSF (общая площадь подвала) - 1stFlrSF (площадь первого этажа): 0.82 (Эти признаки тесно связаны с общей площадью дома и наличием развитой инфраструктуры подвала)
OverallQual - SalePrice: 0.79 (Этот признак имеет одну из самых высоких положительных корреляций с ценой продажи. Как правило, чем выше качество, тем выше и окончательная цена дома)

Для установления зависимости этих переменных используем статистический вывод. Полагаем уровень значимости равным 0.05
Нулевая гипотеза: коэффициент корреляции между переменными X и Y равен нулю.
Во всех вышеперечисленных случаях гипотеза о равенстве нулю коэффициента корреляции не принимается на выбранном уровне значимости.

Применяя t-test для переменных SalePrice и LowQualFinSF (площади с низкокачественной отделкой) можно увидеть следующее: p-value больше 0.05, что предполагает, что любая наблюдаемая связь (или её отсутствие) между SalePrice и LowQualFinSF может быть результатом случайности. То есть нет достаточных доказательств для отклонения нулевой гипотезы, которая утверждает, что нет линейной зависимости между этими двумя переменными.

#### Следующее, что можем проверить это статистически значимое различие между признаками, SalePrice которых находится меньше определенного порога и больше соответственно.
Используем тест Колмогорова-Смирнова для проверки получившихся выборок нормальному распределению. Если это так, то применяется t-критерий Стьюдента, иначе критери Манна-Уитни для проверки равенства медианных значенинй. Почти во всех случаях получаем p_value < 0.05, что говорит о значимом различии признака в двух группах. То есть данный признак достаточно хорошо может помогать в определении стоимости недвижимости.

#### Кластерный анализ.
Используя корреляционную матрицу выберем признаки с наибольшими по абсолютному значениию коэффициентами корреляции. Предполагается, что они наиболее важные: OverallQual, YearBuilt, TotalBsmtSF, 2ndFlrSF, HalfBath, GarageArea, GarageCars.

При использовании метода elbow цель состоит в том, чтобы выбрать количество кластеров таким образом, чтобы добавление еще одного кластера существенно не увеличивало инерцию. По мере увеличения количества кластеров инерция, естественно, будет уменьшаться (каждая точка в среднем будет ближе к своему собственному центру кластера). Однако хорошим выбором часто считается точка, в которой улучшение инерции начинает значительно снижаться.

В данном случае, оптимальным является количество кластеров равное 10.
После обучения модели проводится тест ANOVA для признаков между кластерами. В большинстве случаев видим стат значимое различие в средних одного признака между кластерами. Что говорит о качественном разделении на кластера. Параметры данной модели можно сохранить (Save Model) и использовать для регрессионной модели.


#### 3. Model Building: построение модели и ее интерпретация

Качество подгонки модели. Для построенной модели коэффициент детерминации составляет 0.86, что означает, подогнанная регрессия объясняет большую часть вариабельности зависимой переменной log_SalePrice. Кроме того, p–значение для F–критерия равно 0, следовательно, гипотеза о равенстве нулю всех коэффициентов при признаках отвергается на уровне значимости 0.05, таким образом, регрессия оказывается значимой.

Новая добавленная фича cluster_predictions_0 имеет p_value = 0.259 для t-критерия. Это указывает на то, что, статистически, cluster_predictions_0 не имеет значимого влияния на зависимую переменную при уровне значимости 0.05. Нужно больше экспериментировать с кластеризацией.

Линейность взаимосвязи: среднее значение остатков регрессии близко к нулю, что говорит об истинности предположения о линейности взаимосвязи. Однако есть определенное количество выбросов, которые модель не смогла объяснить.

Нормальность остатков регрессии: На графике (Q-Q normal distribution) точки представляют остатки, а красная линия представляет собой линию, где у нас были бы данные, если они были нормально распределены. Если точки в основном следуют этой линии, то данные могут быть признаны нормально распределенными. В нашем случае это так. Таким образом, модель хорошо объясняет данные. Остатки должны случайны и не проявляют явных закономерностей.

Учитывая, что целевая переменная была предварительно прологарифмирована, выводы о влиянии переменных на зависимую переменную следует интерпретировать как процентное изменение, а не абсолютное. Опишем некоторые из них:

OverallQual (Общее качество)
   - Коэффициент: 0.0894
   - P-значение: 0.000
   - Вывод: Улучшение общего качества на одну единицу приводит к увеличению целевой переменной в среднем на 8.94% (e^0.0894 ≈ 1.0894), предполагая, что все остальные факторы остаются неизменными. Статистическая значимость этого коэффициента высока.

OverallCond (Общее состояние)
   - Коэффициент: 0.0498
   - P-значение: 0.000
   - Вывод: Улучшение общего состояния на одну единицу связано со средним увеличением целевой переменной на 5.12% (e^0.0498 ≈ 1.0512). Эта переменная также статистически значима.

GrLivArea (Жилая площадь)
   - Коэффициент: 0.0001
   - P-значение: 0.000
   - Вывод: Каждый дополнительный квадратный фут жилой площади связан с увеличением целевой переменной на 0.01% (e^0.0001 ≈ 1.0001). Хотя процентное изменение кажется малым, коэффициент статистически значим.