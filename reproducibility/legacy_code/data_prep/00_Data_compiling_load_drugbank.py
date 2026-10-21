#!/usr/bin/env python3

import os
import argparse
import subprocess
from subprocess import CalledProcessError
import pathlib
import zipfile

parser = argparse.ArgumentParser()
db_input = parser.add_mutually_exclusive_group(required=True)
db_input.add_argument(
    '-r', '--dir',
    type=pathlib.Path,
    help='The full path to a directory containing DrugBank csv files.',
    )
db_input.add_argument(
    '-z', '--zipfile',
    type=pathlib.Path,
    help='The full path to a DrugBank zip file.',
    )
mysql_config = parser.add_argument_group('SQL Config')
mysql_config.add_argument(
    '-o', '--host',
    default='127.0.0.1',
    help='The host on which the MySQL server is running. Defaults to localhost.',
    )
mysql_config.add_argument(
    '-t', '--port',
    default='3306',
    help='The port to connect to MySQL. Defaults to 3306.'
)
mysql_config.add_argument(
    '-u', '--user',
    default='root',
    help='The user to connect to MySQL. Defaults to root.',
    )
mysql_config.add_argument(
    '-p', '--password',
    help='The MySQL password for the connecting user.',
    )
mysql_config.add_argument(
    '-d', '--database',
    default='DrugBank',
    help='The database to import data into. Defaults to "DrugBank".',
    )
# Other args
parser.add_argument(
    '--type',
    choices= ['MySQL', 'PostgreSQL'],
    required= True,
    help='The type of SQL files to load.'
)
parser.add_argument(
    '--datadir',
    type=pathlib.Path,
    help='The directory to extract the zip file to. Defaults to the file directory.'
)
parser.add_argument(
    '--drop',
    action='store_true',
    help='Drops the database if it already exists.'
)

class SqlLoader:
    def __init__(self, host, port, user, password, database,
        zipfile=None, filedir=None, datadir=None, drop=False):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.zipfile = zipfile
        self.filedir = filedir
        self.datadir = datadir
        self.drop = drop

    def __call__(self):
        try:
            print("Preparing to load database.")
            self.prep_environment()
            if self.zipfile:
                self.unzip_db_file()
            print("Executing database load.")
            self.load_database()
        except SystemExit:
            print("One or more critical commands failed to run, exiting...")
            raise
        finally:
            self.clean_up()

    def prep_environment(self):
        raise NotImplementedError

    def unzip_db_file(self):
        self._ensure_zip_dir_exists()
        path = self._zipfile_path()
        if not path.is_file():
            raise SystemExit
        target_dir = self._target_zip_dir()
        print(f"Unzipping {path} to {target_dir}")
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)

    def load_database(self):
        workdir = self._sql_file_dir()
        os.chdir(workdir)
        for file in ('create_schema', 'load_tables', 'add_constraints'):
            print(f"Executing sql file: {file}")
            self._load_file(workdir, file)

    def clean_up(self):
        raise NotImplementedError

    def _ensure_zip_dir_exists(self):
        # Ensures target directory exists before file unzipping
        self._target_zip_dir().mkdir(parents=True, exist_ok=True)

    def _target_zip_dir(self):
        return self.datadir.absolute() if self.datadir else self._zipfile_path().parent

    def _zipfile_path(self):
        return self.zipfile.absolute()

    def _sql_file_dir(self):
        return self.filedir.absolute() if self.filedir else self._target_zip_dir()

    def _load_file(self, workdir, filename):
        raise NotImplementedError

    def _sql_args(self):
        args = [
            "--host",
            f"{self.host}",
            "--port",
            f"{self.port}",
            "--user",
            f"{self.user}",
        ]
        self._add_password_arg(args)
        return args

    def _add_password_arg(self, args_array):
        raise NotImplementedError

    def _execute_sql_command(self, command, pipe_command=None, cmd_env=os.environ.copy()):
        try:
            if pipe_command:
                ps1 = subprocess.run(command, check=True, capture_output=True)
                # Pipe shell commands into the sql command
                subprocess.run(pipe_command, input=ps1.stdout, env=cmd_env)
            else:
                subprocess.run(command, env=cmd_env)
        except CalledProcessError:
            cmd_string = ' '.join(command)
            full_cmd = f"{cmd_string} | {' '.join(pipe_command)}" if pipe_command else f"{cmd_string}"
            print(f"Failed to run: {full_cmd}")
            raise SystemExit

class PostgresqlLoader(SqlLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prep_environment(self):
        if self.drop:
            print(f"Dropping existing database {self.database}")
            self._drop_db()
        self._ensure_db_exists()

    def clean_up(self):
        pass

    def _drop_db(self):
        self._execute_sql_command(
            # Double quotes around database name to enforce case sensitivity
            ["echo", f"SELECT 'DROP DATABASE \"{self.database}\"' WHERE EXISTS (SELECT FROM pg_database WHERE datname = '{self.database}')\gexec"],
            # PostgreSQL requires a database for all commands, but it is not possible
            # to drop a database that itself is specified; use "postgres" instead
            pipe_command = self._sql_command_string(specified_db='postgres'),
            cmd_env = self._postgres_command_env(),
        )

    def _ensure_db_exists(self):
        self._execute_sql_command(
            # Double quotes around database name to enforce case sensitivity
            ["echo", f"SELECT 'CREATE DATABASE \"{self.database}\"' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '{self.database}')\gexec"],
            # PostgreSQL requires a database for all commands, but it is not possible
            # to create a database that itself is specified; use "postgres" instead
            pipe_command = self._sql_command_string(specified_db='postgres'),
            cmd_env = self._postgres_command_env(),
        )

    def _load_file(self, workdir, filename):
        sql_path = workdir.joinpath(filename + '.pgsql.sql')
        if filename == 'load_tables':
            # Fix absolute file location in PostgreSQL load statements and switch to \copy
            self._fix_postgresql_load_file(sql_path)
        self._load_sql_file(sql_path)

    def _fix_postgresql_load_file(self, file_path):
        with open(file_path, 'r') as i:
            in_lines = i.readlines()
        with open(file_path, 'w') as o:
            for line in in_lines:
                if line.startswith('SELECT') or line == '\n':
                    # Regular SQL commands within the file don't require any changes
                    o.write(line)
                else:
                    if line.startswith('COPY'):
                        # Replace copy with the psql meta command of the same name
                        # \copy allows for client-side file reading
                        o.write(line.replace(
                            'COPY',
                            '\copy',
                            ).strip('\n') + ' '
                        )
                    elif line.startswith('FROM'):
                        # Fix the absolute file path to coincide with the working dir
                        o.write(line.replace(
                            '/opt/drugbank-data',
                            str(file_path.parent),
                            ).strip('\n') + ' '
                        )
                    elif line.endswith(';\n'):
                        # psql meta commands are terminated by newlines
                        o.write(line.strip(';\n') + '\n')
                    else:
                        # psql meta commands must be on one continuous line as they are
                        # terminated by a newline; fold up subsequent lines onto one line
                        o.write(line.strip().strip('\n') + ' ')

    def _load_sql_file(self, file_handle):
        sql_cmd = self._sql_command_string()
        sql_cmd.extend(['-f', file_handle])
        self._execute_sql_command(sql_cmd, cmd_env=self._postgres_command_env())

    def _sql_command_string(self, specified_db=None):
        sql_cmd = ['psql']
        sql_cmd.extend(self._sql_args())
        # PostgreSQL always requires a database in an SQL command
        target = specified_db if specified_db else self.database
        sql_cmd.extend(['--dbname', f"{target}"])
        return sql_cmd

    def _postgres_command_env(self):
        if self.password:
            return dict(os.environ.copy(), PGPASSWORD=f"{self.password}")
        return os.environ.copy()

    def _add_password_arg(self, args_array):
        # Explicit pass for PostgreSQL: passwords for PostgreSQL need to be set
        # in an env variable to avoid having to enter them on the command line
        pass

class MysqlLoader(SqlLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prep_environment(self):
        # Ensure server allows for local file loading
        self._set_server_local_infile()
        # Disable strict foreign key constraints
        self._set_foreign_key_checks(False)
        if self.drop:
            print(f"Dropping existing database {self.database}")
            self._drop_db()
        self._ensure_db_exists()

    def clean_up(self):
        # Ensure local file loading is turned off again
        self._set_server_local_infile(False)
        # Turn back on foreign key checks
        self._set_foreign_key_checks()

    def _set_server_local_infile(self, local=True):
        self._execute_sql_command(
            ["echo", f"SET GLOBAL local_infile = {local}"],
            pipe_command = self._sql_command_string(),
        )

    def _set_foreign_key_checks(self, check=True):
        self._execute_sql_command(
            ["echo", f"SET GLOBAL FOREIGN_KEY_CHECKS = {check}"],
            pipe_command = self._sql_command_string(),
        )

    def _drop_db(self):
        self._execute_sql_command(
            ["echo", f"DROP DATABASE IF EXISTS {args.database}"],
            pipe_command = self._sql_command_string(),
        )

    def _ensure_db_exists(self):
        self._execute_sql_command(
            ["echo", f"CREATE DATABASE IF NOT EXISTS {args.database}"],
            pipe_command = self._sql_command_string(),
        )

    def _load_file(self, workdir, filename):
        sql_path = workdir.joinpath(filename + '.mysql.sql')
        self._execute_sql_command(
            # MySQL allows to cat the file contents directly to the command line
            ['cat', sql_path],
            pipe_command = self._sql_command_string(self.database)
        )

    def _sql_command_string(self, specified_db=None):
        sql_cmd = ['mysql']
        sql_cmd.extend(self._sql_args())
        if specified_db:
            # For MySQL, we only need to specify a database if we're executing commands to it
            sql_cmd.append('--local-infile')    # Allows for local file loading
            sql_cmd.append(specified_db)
        return sql_cmd

    def _add_password_arg(self, args_array):
        if self.password:
            args_array.append(f"--password={self.password}")

def validate_args():
    target_dir = args.datadir
    if target_dir and not args.zipfile:
        print("Cannot specify a data directory without a zip file.")
        raise SystemExit
    if target_dir and target_dir.is_file():
        print("Specified data directory is not a directory.")
        raise SystemExit

def get_loader_class(sql_type):
    return MysqlLoader if sql_type == 'MySQL' else PostgresqlLoader

def main():
    try:
        validate_args()
    except SystemExit:
        print("Script run with bad arguments, please consult -h/--help and try again.")
        raise
    # Main execution
    loader_class = get_loader_class(args.type)
    loader = loader_class(
        host = args.host,
        port = args.port,
        user = args.user,
        password = args.password,
        database = args.database,
        zipfile = args.zipfile,
        filedir = args.dir,
        datadir = args.datadir,
        drop = args.drop,
    )
    # May raise SystemExit
    loader()

if __name__ == '__main__':
    args = parser.parse_args()
    main()