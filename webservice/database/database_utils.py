import sqlite3 as sql

DATABASE_PATH = '/home/twokey/PycharmProjects/KsDynAUTH/webservice/database/database.db'

def create_db() -> None:
    """
    Creates the users table in the database if it doesn't exist.
    """
    try:
        conn = sql.connect(DATABASE_PATH)
        conn.execute(
            'CREATE TABLE IF NOT EXISTS users('
            'id INTEGER PRIMARY KEY AUTOINCREMENT, '
            'username TEXT UNIQUE NOT '
            'NULL, password TEXT NOT NULL)')
        conn.close()
    except Exception as e:
        print(e)


def test_connection_db() -> None:
    """
    Tests the connection to the database.
    """
    try:
        _ = sql.connect(DATABASE_PATH)
    except Exception as e:
        print(e)
        exit()


def drop_db() -> None:
    """
    Drops the users table from the database.
    """
    try:
        conn = sql.connect(DATABASE_PATH)
        conn.execute('DROP TABLE users')
        conn.close()
    except Exception as e:
        print(e)


def add_user_and_passw(username: str, password: str) -> tuple:
    """
    Adds a user with username and password to the users table.

    Args:
        username (str): The username of the user.
        password (str): The password of the user.

    Returns:
        tuple: A tuple containing the user ID and a boolean indicating if the user was added successfully.
    """
    try:
        with sql.connect(DATABASE_PATH) as con:
            cursor = con.cursor()
            cursor.execute('SELECT * FROM users where username=?', [username])
            con.row_factory = sql.Row
            rows = cursor.fetchall()
            if rows:
                return 0, False
            else:
                con.commit()
                cursor.execute('INSERT INTO users (username, password)  VALUES (?, ?)', (username, password))
                user_id = cursor.lastrowid
                return user_id, True
    except Exception as e:
        print(e)


def check_user_and_passw(username: str, password: str) -> tuple:
    """
    Checks if the username and password match a user in the users table.

    Args:
        username (str): The username of the user.
        password (str): The password of the user.

    Returns:
        tuple: A tuple containing the status code, a boolean indicating if the credentials are valid, and the user ID.
    """
    try:
        with sql.connect(DATABASE_PATH) as con:
            cursor = con.cursor()
            cursor.execute(
                'SELECT * FROM users WHERE username=?',
                [username])
            con.row_factory = sql.Row
            rows = cursor.fetchall()
            if rows:
                for row in rows:
                    user_id = row[0]
                    passw = row[2]
                if passw == password:
                    return 2, True, user_id
                else:
                    return 1, False, user_id
            else:
                return 3, False, 0
    except Exception as e:
        print(e)
        return None, False, None


def get_user_and_passw(user_id: int) -> tuple:
    """
    Retrieves the username and password of a user based on the user ID.

    Args:
        user_id (int): The ID of the user.

    Returns:
        tuple: A tuple containing the username and password of the user.
    """
    try:
        user_id -= 1
        con = sql.connect(DATABASE_PATH)
        con.row_factory = sql.Row
        cursor = con.cursor()
        cursor.execute('SELECT * FROM users')

        rows = cursor.fetchall()
        username = rows[int(user_id)]['username']
        password = rows[int(user_id)]['password']
        return username, password
    except Exception as e:
        print(e)


def get_user_id(username: str) -> int:
    """
        Retrieves the user ID based on the username.

        Args:
            username (str): The username of the user.

        Returns:
            int: The ID of the user.
        """
    try:
        con = sql.connect(DATABASE_PATH)
        con.row_factory = sql.Row
        cursor = con.cursor()
        cursor.execute('SELECT * FROM users WHERE username=?', (username,))

        rows = cursor.fetchall()
        user_id = None
        if rows:
            for row in rows:
                user_id = row[0]
        return user_id
    except Exception as e:
        print(e)
