from aiogram import Bot, Dispatcher, executor, types
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
import pyodbc
import hashlib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

API_TOKEN = '6218949302:AAFqBOsQE76s7j3eVVlNLcaRezSVQeU7uhU'

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

cnxn = pyodbc.connect("Driver={SQL Server Native Client 11.0};"
                      "Server=RANELPC;"
                      "Database=recruterra;"
                      "Trusted_Connection=yes;")

cursor = cnxn.cursor()


@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    await message.reply("Привет!\n\nАвторизуйтесь пожалуйста в системе:\n\n/auth [ваш логин] [ваш пароль]")


@dp.message_handler(commands=("auth"), commands_prefix="/")
async def send_enter_login(message: types.Message):
    value = message.text
    value = value[6:]
    login, password = value.split(' ')
    cursor.execute('SELECT * FROM Administrator')
    admins = cursor.fetchall()
    for admin in admins:
        if admin.TelegramId == message.from_user.id:
            await message.answer("Вы уже проходили авторизацию!")
            break
    else:
        cursor.execute('SELECT * FROM Users')
        users = cursor.fetchall()
        for user in users:
            if user.Login == login and user.Password == password_hashing(password) and user.Role == "Администратор":
                cursor.execute(
                    'INSERT INTO Administrator (Id, TelegramId) VALUES (?, ?)', (user.Id, message.from_user.id))
                cnxn.commit()
                await message.answer("Авторизация прошла успешно!")


@dp.message_handler(commands=['profile'])
async def send_profile(message: types.Message):
    if get_auth_status(message.from_user.id) == True:
        cursor.execute(
            f'SELECT * FROM Administrator WHERE TelegramId = {message.from_user.id}')
        admins = cursor.fetchall()
        for adm in admins:
            await message.answer(f"Ваш логин: {find_user_by_id(adm.Id)}!")
    else:
        await message.answer("Авторизуйтесь!")


@dp.message_handler(commands=['employers'])
async def send_vacancies(message: types.Message):
    if get_auth_status(message.from_user.id) == True:
        cursor.execute('SELECT * FROM Employers')
        employers = cursor.fetchall()
        for employer in employers:
            if employer.IsApproved == 0:
                employersacceptkb = InlineKeyboardMarkup(row_width=2)
                employersacceptbutton = InlineKeyboardButton(
                    text='Принять', callback_data=f'acceptemployers_{employer.Id}')
                employersdissmisbutton = InlineKeyboardButton(
                    text='Отклонить', callback_data=f'dismissemployers_{employer.Id}')
                employersacceptkb.add(
                    employersacceptbutton, employersdissmisbutton)
                await message.answer(f"Наименование: {employer.CompanyName}\n\nГород: {find_city_by_id(employer.IdCity)}\nДата регистрации компании: {employer.CreationDate.strftime('%d.%m.%Y')}\nОГРН {employer.MSRN}\nАдрес: ул. {employer.Street}, д. {employer.House}, п. {employer.Apartment}", reply_markup=employersacceptkb)
    else:
        await message.answer("Авторизуйтесь!")


@dp.message_handler(commands=['vacancies'])
async def send_algorithm(message: types.Message):
    if get_auth_status(message.from_user.id) == True:
        cursor.execute('SELECT * FROM Vacancies')
        vacancies = cursor.fetchall()
        for vacancy in vacancies:
            if vacancy.IsConfirmed == 0:
                fake_job = {'Title': str(vacancy.Position),
                            'Salary': str(vacancy.Salary),
                            'Description': str(vacancy.Description),
                            'Requirements': str(vacancy.Obligations),
                            'Conditions': str(find_typeemp_by_id(vacancy.IdTypeOfEmployment)),
                            'WorkEx': set_work_ex(vacancy.WorkExperience),
                            'Exist': False}
                vacancyacceptkb = InlineKeyboardMarkup(row_width=2)
                vacancyacceptbutton = InlineKeyboardButton(
                    text='Принять', callback_data=f'accept_{vacancy.Id}')
                vacancydissmisbutton = InlineKeyboardButton(
                    text='Отклонить', callback_data=f'dismiss_{vacancy.Id}')
                vacancyacceptkb.add(vacancyacceptbutton, vacancydissmisbutton)
                set_optimal_salary(fake_job, vacancy.Id)
                await message.answer(f"{vacancy.Position}\n\n{vacancy.Description}\n{vacancy.Salary} руб.\n\nАлгоритм, который использует метод случайного леса предполагает, что {jobRandomForestClassifier(fake_job)}\n\nАлгоритм, который использует метод логической регрессии предполагает, что {jobLogisticRegression(fake_job)}", reply_markup=vacancyacceptkb)
    else:
        await message.answer("Авторизуйтесь!")


def set_work_ex(workex):
    if workex == 0:
        return "Без опыта"
    elif workex == 1:
        return "Опыт от 1-го года"
    elif workex == 2:
        return "Опыт от 2-х лет"
    elif workex == 3:
        return "Опыт от 3-х лет"
    elif workex == 4:
        return "Опыт от 4-х лет"
    elif workex == 5:
        return "Опыт от 5-ти лет"
    elif workex == 6:
        return "Опыт от 6-ти лет"


def set_optimal_salary(fake_job, vacancy_id):
    salary = int(salaryRandomForestRegressor(fake_job))
    cursor.execute(
        f'UPDATE Vacancies SET OptimalSalary={salary} WHERE Id={vacancy_id}')
    cnxn.commit()


def find_user_by_id(id):
    for adm in cursor.execute('SELECT Id, Login FROM Users'):
        if adm.Id == id:
            return adm.Login


def find_typeemp_by_id(id):
    for typeemp in cursor.execute('SELECT Id, Type FROM TypeOfEmployments'):
        if typeemp.Id == id:
            return typeemp.Type


def find_city_by_id(id):
    for city in cursor.execute('SELECT Id, Name FROM Cities'):
        if city.Id == id:
            return city.Name


def get_auth_status(id):
    cursor.execute(
        f'SELECT * FROM Administrator WHERE TelegramId = {id}')
    admins = cursor.fetchall()
    for admin in admins:
        return True


def get_vacancy_by_id(id):
    cursor.execute(
        f'SELECT * FROM Vacancies WHERE Id = {id}')
    vacancies = cursor.fetchall()
    for vacancy in vacancies:
        return vacancy.Position


def get_employer_by_id(id):
    cursor.execute(
        f'SELECT * FROM Employers WHERE Id = {id}')
    employers = cursor.fetchall()
    for employer in employers:
        return employer.CompanyName


def password_hashing(s):
    sha256 = hashlib.sha256()
    sha256.update(s.encode('utf-8'))
    return sha256.hexdigest()


@dp.callback_query_handler(lambda c: c.data and c.data.startswith(('accept_', 'dismiss_')))
async def handle_callback_query(callback_query: types.CallbackQuery):
    command, vacancy_id = callback_query.data.split('_')
    vacancy = get_vacancy_by_id(vacancy_id)
    if command == 'accept':
        cursor.execute(
            f'UPDATE Vacancies SET IsConfirmed=1 WHERE Id={vacancy_id}')
        cnxn.commit()
        await bot.answer_callback_query(
            callback_query.id,
            text=f"Вакансия {vacancy} была подтверждена", show_alert=True)
        await bot.edit_message_reply_markup(
            chat_id=callback_query.from_user.id,
            message_id=callback_query.message.message_id,
            reply_markup=None
        )
    elif command == 'dismiss':
        await bot.answer_callback_query(
            callback_query.id,
            text=f"Вакансия {vacancy} была отклонена", show_alert=True)
        cursor.execute(
            f'UPDATE Vacancies SET IsConfirmed=0 WHERE Id={vacancy_id}')
        cnxn.commit()
        await bot.edit_message_reply_markup(
            chat_id=callback_query.from_user.id,
            message_id=callback_query.message.message_id,
            reply_markup=None
        )


@dp.callback_query_handler(lambda c: c.data and c.data.startswith(('acceptemployers_', 'dismissemployers_')))
async def handle_callback_query(callback_query: types.CallbackQuery):
    command, employer_id = callback_query.data.split('_')
    employer = get_employer_by_id(employer_id)
    if command == 'acceptemployers':
        await bot.answer_callback_query(
            callback_query.id,
            text=f"Работодатель {employer} был подтвержден", show_alert=True)
        cursor.execute(
            f'UPDATE Employers SET IsApproved=1 WHERE Id={employer_id}')
        cnxn.commit()
        await bot.edit_message_reply_markup(
            chat_id=callback_query.from_user.id,
            message_id=callback_query.message.message_id,
            reply_markup=None
        )
    elif command == 'dismissemployers':
        await bot.answer_callback_query(
            callback_query.id,
            text=f"Работодатель {employer} не прошел проверку", show_alert=True)
        cursor.execute(
            f'UPDATE Employers SET IsApproved=0 WHERE Id={employer_id}')
        cnxn.commit()
        await bot.edit_message_reply_markup(
            chat_id=callback_query.from_user.id,
            message_id=callback_query.message.message_id,
            reply_markup=None
        )


@dp.message_handler()
async def echo(message: types.Message):
    await message.answer("Воспользуйтесь меню!")


def jobLogisticRegression(fake_job):
    # Загрузка данных
    data = pd.read_csv('dataset.csv', sep=';', header=None)
    data.columns = ['Title', 'Salary', 'Description',
                    'Requirements', 'Conditions', 'WorkEx', 'Exist']

    # Добавление фейковой вакансии в данные
    data = data._append(fake_job, ignore_index=True)

    # Создание корпуса текстов
    corpus = data['Title'] + ' ' + data['Salary'] + ' ' + data['Description'] + \
        ' ' + data['Requirements'] + ' ' + \
        data['Conditions'] + ' ' + data['WorkEx']

    # Создание матрицы TF-IDF признаков
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)

    # Обучение модели логистической регрессии
    y = data['Exist']
    clf = LogisticRegression()
    clf.fit(X[:-1], y[:-1])

    # Предсказание результата для новой вакансии
    fake_job_vector = vectorizer.transform(
        [fake_job['Title'] + ' ' + fake_job['Salary'] + fake_job['Description'] + ' ' + fake_job['Requirements'] + ' ' + fake_job['Conditions'] + ' ' + fake_job['WorkEx']])
    fake_job_prob = clf.predict_proba(fake_job_vector)[0][1]

    # Вывод результата
    if fake_job_prob > 0.2:
        result = f'вакансия является ненастоящей ({round((1 - fake_job_prob) * 100, 3)}%)'
        return result
    else:
        result = f'вакансия является настоящей ({round((1 - fake_job_prob) * 100, 3)}%)'
        return result


def jobRandomForestClassifier(fake_job):
    data = pd.read_csv('dataset.csv', sep=';', header=None)
    data.columns = ['Title', 'Salary', 'Description',
                    'Requirements', 'Conditions', 'WorkEx', 'Exist']

    data = data._append(fake_job, ignore_index=True)
    corpus = data['Title'] + ' ' + data['Salary'] + ' ' + data['Description'] + \
        ' ' + data['Requirements'] + ' ' + \
        data['Conditions'] + ' ' + data['WorkEx']
    # stop_words = nltk.corpus.stopwords.words('russian')
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    y = data['Exist']
    clf = RandomForestClassifier()
    clf.fit(X[:-1], y[:-1])
    fake_job_vector = vectorizer.transform([fake_job['Title'] + ' ' + fake_job['Salary'] + fake_job['Description'] +
                                           ' ' + fake_job['Requirements'] + ' ' + fake_job['Conditions'] + ' ' + fake_job['WorkEx']])
    fake_job_prob = clf.predict_proba(fake_job_vector)[0][1]
    if fake_job_prob > 0.2:
        result = f'вакансия является ненастоящей ({round((1 - fake_job_prob) * 100, 3)}%)'
        return result
    else:
        result = f'вакансия является настоящей ({round((1 - fake_job_prob) * 100, 3)}%)'
        return result


def salaryRandomForestRegressor(new_job):  # Загрузка данных
    data = pd.read_csv('dataset.csv', sep=';', skiprows=[0], header=None)
    data.columns = ['Title', 'Salary', 'Description',
                    'Requirements', 'Conditions', 'WorkEx', 'Exist']
    corpus = data['Title'] + ' ' + data['Description'] + \
        ' ' + data['Requirements'] + ' ' + \
        data['Conditions'] + ' ' + data['WorkEx']
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    y = data['Salary']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    rf = RandomForestRegressor(n_estimators=500, random_state=42)
    rf.fit(X_train, y_train)
    # y_pred = rf.predict(X_test)
    # rmse = mean_squared_error(y_test, y_pred, squared=False)
    # print(f'Среднеквадратичная ошибка: {rmse}')

    new_job_vector = vectorizer.transform([new_job['Title'] + ' ' + new_job['Description'] +
                                          ' ' + new_job['Requirements'] + ' ' + new_job['Conditions'] + ' ' + new_job['WorkEx']])
    optimal_salary = rf.predict(new_job_vector)[0]
    return optimal_salary


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
