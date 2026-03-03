import peewee
import os
import subprocess
from typing import Self
from ..cli import register_command
from ..input import InputLoader, InputAcceptable
import sys
data_base_path = os.environ.get("BINDGAR_SLURM_DATA_BASE", "~/.bindgar_slurm_data_base.db")

# define the enum for the status of the job
class JobStatus:
    PENDING = 'pending'
    RUNNING = 'running'
    FINISHED = 'finished'
    UNFINISHED = 'unfinished'
    FAILED = 'failed'
    CHECK_REQUIRED = 'check_required'

# A model to store an configuration set of a batch of slurm job.
class ConfigurationSet(peewee.Model):
    config_name = peewee.CharField(unique=True)
    slurm_file_path = peewee.CharField()
    queue = peewee.CharField()
    ntasks = peewee.IntegerField()
    cpu_per_task = peewee.IntegerField()
    module_tasks = peewee.CharField()
    env_sets = peewee.CharField(null=True) # the environment variables to set before running the job, separated by ';', for example: "export PATH=/path/to/bin:$PATH; export LD_LIBRARY_PATH=/path/to/lib:$LD_LIBRARY_PATH"
    @classmethod
    def new_configuration_set(cls) -> Self:
        # open the vim editor for user to input the configuration set information, and then create a new configuration set in the database.
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".tmp") as tmp:
            tmp.write(b"# Please input the configuration set information in the following format:\n")
            tmp.write(b"# config_name: a unique name for this configuration set\n")
            tmp.write(b"# slurm_file_path: the path to store the generated slurm files\n")
            tmp.write(b"# queue: the queue to submit the jobs\n")
            tmp.write(b"# ntasks: the number of tasks for each job\n")
            tmp.write(b"# cpu_per_task: the number of cpu per task for each job\n")
            tmp.write(b"# module_tasks: the module tasks before running the job, separated by ';', do not input 'module' at the begaining\n")
            tmp.write(b"#     use `purge`, `load <module_name>`, not `module purge`, `module load <module_name>`\n")
            # write the default configuration set information for user to edit
            tmp.write(b"config_name: default_config\n")
            tmp.write(b"slurm_file_path: /path/to/slurm/files\n")
            tmp.write(b"queue: default\n")
            tmp.write(b"ntasks: 1\n")
            tmp.write(b"cpu_per_task: 1\n")
            tmp.write(b"module_tasks: load python/anaconda/2020.11\n")
            tmp.write(b"env_sets: export OMP_NUM_THREADS=1;\n")
            tmp.flush()
            subprocess.run(['vim', tmp.name])
            with open(tmp.name, 'r') as f:
                lines = f.readlines()
                config_dict = {}
                for line in lines:
                    line = line.strip()
                    if line.startswith("#") or line == "":
                        continue
                    key, value = line.split(":", 1)
                    config_dict[key.strip()] = value.strip()
                config_set = cls.create(
                    config_name=config_dict['config_name'],
                    slurm_file_path=config_dict['slurm_file_path'],
                    queue=config_dict['queue'],
                    ntasks=int(config_dict['ntasks']),
                    cpu_per_task=int(config_dict['cpu_per_task']),
                    module_tasks=config_dict['module_tasks'],
                    env_sets=config_dict.get('env_sets', None)
                )
                return config_set
    def edit_configuration_set(self) -> None:
        # open the vim editor for user to edit the configuration set information, and then update the configuration set in the database.
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".tmp") as tmp:
            tmp.write(b"# Please edit the configuration set information in the following format:\n")
            tmp.write(b"# config_name: a unique name for this configuration set\n")
            tmp.write(b"# slurm_file_path: the path to store the generated slurm files\n")
            tmp.write(b"# queue: the queue to submit the jobs\n")
            tmp.write(b"# ntasks: the number of tasks for each job\n")
            tmp.write(b"# cpu_per_task: the number of cpu per task for each job\n")
            tmp.write(b"# module_tasks: the module tasks before running the job, separated by ';', do not input 'module' at the begaining\n")
            tmp.write(b"#     use `purge`, `load <module_name>`, not `module purge`, `module load <module_name>`\n")
            tmp.write(b"# env_sets: the environment variables to set before running the job, separated by ';', for example: \"export PATH=/path/to/bin:$PATH; export LD_LIBRARY_PATH=/path/to/lib:$LD_LIBRARY_PATH\"\n")
            # write the current configuration set information for user to edit
            tmp.write(f"config_name: {self.config_name}\n".encode('utf-8'))
            tmp.write(f"slurm_file_path: {self.slurm_file_path}\n".encode('utf-8'))
            tmp.write(f"queue: {self.queue}\n".encode('utf-8'))
            tmp.write(f"ntasks: {self.ntasks}\n".encode('utf-8'))
            tmp.write(f"cpu_per_task: {self.cpu_per_task}\n".encode('utf-8'))
            tmp.write(f"module_tasks: {self.module_tasks}\n".encode('utf-8'))
            tmp.write(f"env_sets: {self.env_sets}\n".encode('utf-8'))
            tmp.flush()
            subprocess.run(['vim', tmp.name])
            with open(tmp.name, 'r') as f:
                lines = f.readlines()
                config_dict = {}
                for line in lines:
                    line = line.strip()
                    if line.startswith("#") or line == "":
                        continue
                    key, value = line.split(":", 1)
                    config_dict[key.strip()] = value.strip()
                self.config_name = config_dict['config_name']
                self.slurm_file_path = config_dict['slurm_file_path']
                self.queue = config_dict['queue']
                self.ntasks = int(config_dict['ntasks'])
                self.cpu_per_task = int(config_dict['cpu_per_task'])
                self.module_tasks = config_dict['module_tasks']
                self.env_sets = config_dict.get('env_sets', None)
                self.save()

# A model to store the slurm job informations: The dir job is running in, the job id in slurm system, the status of the job (finished, unfinished, failed), the re-submit script if the job is failed, the re-submit script if the job is unfinished due to time limit.
class SlurmJob(peewee.Model):
    # job_id is the primary key of the table, and it should be set automatically by the database.
    job_id = peewee.AutoField() 
    current_slurm_id = peewee.CharField(unique=True)
    job_dir = peewee.CharField()
    status = peewee.CharField()
    resubmit_script_failed = peewee.CharField(null=True)
    resubmit_script_unfinished = peewee.CharField(null=True)
    unfinished_times = peewee.IntegerField(default=0)
    failed_times = peewee.IntegerField(default=0)
    max_unfinished_times = peewee.IntegerField(default=3)
    max_failed_times = peewee.IntegerField(default=3)
    finished_flag_file = peewee.CharField(default="finished") # if the job is finished, there should be a file in the job dir, and the last line of the file should be "1". This file is managed by the slurm task itself.
    error_flag_file = peewee.CharField(default="failed") # if the job is failed, there should be a file in the job dir, and the last line of the file should be "-1", else it should be "0". This file is managed by the slurm task itself.
    configure_set = peewee.ForeignKeyField(ConfigurationSet, backref='jobs')
    
    @classmethod
    def register_job(cls, configure_set_name: str, 
                     job_dir: str, 
                     current_slurm_id: str, 
                     status: str, 
                     resubmit_script_failed: str|None = None, 
                     resubmit_script_unfinished: str|None = None, 
                     max_unfinished_times: int = 3, 
                     max_failed_times: int = 3,
                     error_flag_file: str = "is_failed",
                     finished_flag_file: str = "is_finished"
                     ) -> Self:
        configure = ConfigurationSet.get(ConfigurationSet.config_name == configure_set_name)
        current_slurm_id.strip('_')
        job = SlurmJob.create(
            job_dir=job_dir,
            current_slurm_id=current_slurm_id,
            status=status,
            resubmit_script_failed=resubmit_script_failed,
            resubmit_script_unfinished=resubmit_script_unfinished,
            max_unfinished_times=max_unfinished_times,
            max_failed_times=max_failed_times,
            error_flag_file=error_flag_file,
            finished_flag_file=finished_flag_file,
            configure_set=configure
        )
        return job
    
    def check_slurm_status(self, slurm_id: str) -> str:
        # check the status of the job in slurm system using sacct command
        import subprocess
        result = subprocess.run(['sacct', '-j', slurm_id, '--format=State', '--noheader','-X'], stdout=subprocess.PIPE)
        # It should be note that the status returned by sacct command is not the same as the status we defined in JobStatus, we need to map them to our status.
        status = result.stdout.decode('utf-8').strip()
        if status in ['PENDING', 'CONFIGURING', 'SUSPENDED']:
            return JobStatus.PENDING
        elif status in ['RUNNING', 'COMPLETING']:
            return JobStatus.RUNNING
        elif status in ['COMPLETED','FAILED', 'CANCELLED', 'TIMEOUT', 'PREEMPTED','CANCELLED+']:
            # check the finished flag file to determine whether the job is finished or unfinished
            finished_flag = None
            failed_flag = None
            failed_flag_file = os.path.join(str(self.job_dir).strip(), str(self.error_flag_file).strip())
            finished_flag_file = os.path.join(str(self.job_dir).strip(), str(self.finished_flag_file).strip())
            if os.path.exists(finished_flag_file):
                with open(finished_flag_file, 'r') as f:
                    finished_flag = f.read().strip()
            if os.path.exists(failed_flag_file):
                with open(failed_flag_file, 'r') as f:
                    failed_flag = f.read().strip()
            if finished_flag == "1":
                return JobStatus.FINISHED
            elif failed_flag == "-1":
                return JobStatus.FAILED
            else:
                return JobStatus.UNFINISHED
        else:
            print(f"Unknown status {status} for slurm job {slurm_id}, please check the job status manually.")    
            return JobStatus.CHECK_REQUIRED
    def resubmit_failed_task(self) -> str:
        return self._resubmit_task(
            script_type="failed", 
            script=self.resubmit_script_failed, # type: ignore
            times=self.failed_times # type: ignore
        )

    def resubmit_unfinished_task(self) -> str:
        return self._resubmit_task(
            script_type="unfinished", 
            script=self.resubmit_script_unfinished, # type: ignore
            times=self.unfinished_times # type: ignore
        )

    def _resubmit_task(self, script_type: str, script: str, times: int) -> str:
        if script is None:
            raise ValueError(f"No resubmit script for {script_type} job {self.job_id:08x}, "
                            f"please check the job manually.")
        
        # 生成新的slurm文件
        os.makedirs(self.configure_set.slurm_file_path, exist_ok=True)
        new_slurm_file_path = os.path.join(
            self.configure_set.slurm_file_path, 
            f"{self.job_id:08x}_resubmit_{script_type}_{times}.slurm"
        )
        
        # 写入slurm文件内容
        with open(new_slurm_file_path, 'w') as f:
            self._write_slurm_header(f, script_type, times)
            self._write_module_and_env(f)
            self._write_work_directory(f)
            self._write_execution_script(f, script_type, script, times)
        
        # 提交作业并返回新的slurm ID
        result = subprocess.run(
            ['sbatch', new_slurm_file_path], 
            stdout=subprocess.PIPE, 
            cwd=self.configure_set.slurm_file_path
        )
        new_slurm_id = result.stdout.decode('utf-8').strip().split()[-1]
        return new_slurm_id

    def _write_slurm_header(self, f, script_type: str, times: int) -> None:
        """写入SLURM脚本的头部信息"""
        f.write("#!/bin/bash\n")
        f.write(f"#SBATCH --job-name={self.job_id:08x}_resubmit_{script_type}_{times}\n")
        f.write(f"#SBATCH --output={self.job_id:08x}_resubmit_{script_type}_{times}.out\n")
        f.write(f"#SBATCH --error={self.job_id:08x}_resubmit_{script_type}_{times}.err\n")
        f.write(f"#SBATCH --ntasks={self.configure_set.ntasks}\n")
        f.write(f"#SBATCH --cpus-per-task={self.configure_set.cpu_per_task}\n")
        f.write(f"#SBATCH --partition={self.configure_set.queue}\n")

    def _write_module_and_env(self, f) -> None:
        """写入模块加载和环境变量设置"""
        if self.configure_set.module_tasks:
            module_tasks = self.configure_set.module_tasks.split(';')
            for module_task in module_tasks:
                f.write(f"module {module_task}\n")
        
        if self.configure_set.env_sets:
            env_sets = self.configure_set.env_sets.split(';')
            for env_set in env_sets:
                f.write(f"{env_set}\n")

    def _write_work_directory(self, f) -> None:
        """写入工作目录切换"""
        f.write(f"cd {self.job_dir}\n")

    def _write_execution_script(self, f, script_type: str, script: str, times: int) -> None:
        """写入执行脚本和错误处理逻辑"""
        f.write(f"trap 'echo 0 > {self.finished_flag_file}; echo -1 > {self.error_flag_file}' EXIT\n")
        f.write(f"{script} > {self.job_dir}/resubmit_{script_type}_{times}.log 2>&1\n")
        
        f.write(f"if [ $? -eq 0 ]; then\n")
        f.write(f"    echo 1 > {self.finished_flag_file}\n")
        f.write(f"    echo 0 > {self.error_flag_file}\n")
        f.write(f"    trap - EXIT\n")
        f.write(f"else\n")
        f.write(f"    echo 0 > {self.finished_flag_file}\n")
        f.write(f"    echo -1 > {self.error_flag_file}\n")
        f.write(f"fi\n")
    def _ref_resubmit_failed_task(self) -> str: # no longer used, but keep it for reference
        if self.resubmit_script_failed is not None:
            # generate a new slurm file 
            if not os.path.exists(self.configure_set.slurm_file_path):
                os.makedirs(self.configure_set.slurm_file_path)
            new_slurm_file_path = os.path.join(self.configure_set.slurm_file_path, f"{self.job_id:08x}_resubmit_failed_{self.failed_times}.slurm")
            with open(new_slurm_file_path, 'w') as f:
                f.write("#!/bin/bash\n")
                f.write(f"#SBATCH --job-name={self.job_id:08x}_resubmit_failed_{self.failed_times}\n")
                f.write(f"#SBATCH --output={self.job_id:08x}_resubmit_failed_{self.failed_times}.out\n")
                f.write(f"#SBATCH --error={self.job_id:08x}_resubmit_failed_{self.failed_times}.err\n")
                f.write(f"#SBATCH --ntasks={self.configure_set.ntasks}\n")
                f.write(f"#SBATCH --cpus-per-task={self.configure_set.cpu_per_task}\n")
                f.write(f"#SBATCH --partition={self.configure_set.queue}\n")
                module_tasks = self.configure_set.module_tasks.split(';')
                for module_task in module_tasks:
                    f.write(f"module {module_task}\n")
                if self.configure_set.env_sets is not None:
                    env_sets = self.configure_set.env_sets.split(';')
                    for env_set in env_sets:
                        f.write(f"{env_set}\n")
                f.write(f"cd {self.job_dir}\n")
                # include the resubmit script with a trap to handle unexpected exits
                f.write(f"trap 'echo 0 > {self.finished_flag_file}; echo -1 > {self.error_flag_file}' EXIT\n")
                f.write(f"{self.resubmit_script_failed} > {self.job_dir}/resubmit_failed_{self.failed_times}.log 2>&1\n")
                f.write(f"if [ $? -eq 0 ]; then\n")
                f.write(f"    echo 1 > {self.finished_flag_file}\n")
                f.write(f"    echo 0 > {self.error_flag_file}\n")
                f.write(f"    trap - EXIT\n")  # Disable the trap if the script succeeds
                f.write(f"else\n")
                f.write(f"    echo 0 > {self.finished_flag_file}\n")
                f.write(f"    echo -1 > {self.error_flag_file}\n")
                f.write(f"fi\n")
            # submit the new slurm file and return the new slurm id
            # cd the slurm file path before submitting the job, to avoid the problem that the slurm file path is too long.
            result = subprocess.run(['sbatch', new_slurm_file_path], stdout=subprocess.PIPE, cwd=self.configure_set.slurm_file_path)
            new_slurm_id = result.stdout.decode('utf-8').strip().split()[-1]
            return new_slurm_id
        else:
            raise ValueError(f"No resubmit script for failed job {self.job_id:08x}, please check the job manually.")
    
    def _ref_resubmit_unfinished_task(self) -> str: # no longer used, but keep it for reference
        if self.resubmit_script_unfinished is not None:
            # generate a new slurm file 
            if not os.path.exists(self.configure_set.slurm_file_path):
                os.makedirs(self.configure_set.slurm_file_path)
            new_slurm_file_path = os.path.join(self.configure_set.slurm_file_path, f"{self.job_id:08x}_resubmit_unfinished_{self.unfinished_times}.slurm")
            with open(new_slurm_file_path, 'w') as f:
                f.write("#!/bin/bash\n")
                f.write(f"#SBATCH --job-name={self.job_id:08x}_resubmit_unfinished_{self.unfinished_times}\n")
                f.write(f"#SBATCH --output={self.job_id:08x}_resubmit_unfinished_{self.unfinished_times}.out\n")
                f.write(f"#SBATCH --error={self.job_id:08x}_resubmit_unfinished_{self.unfinished_times}.err\n")
                f.write(f"#SBATCH --ntasks={self.configure_set.ntasks}\n")
                f.write(f"#SBATCH --cpus-per-task={self.configure_set.cpu_per_task}\n")
                f.write(f"#SBATCH --partition={self.configure_set.queue}\n")
                module_tasks = self.configure_set.module_tasks.split(';')
                for module_task in module_tasks:
                    f.write(f"module {module_task}\n")
                if self.configure_set.env_sets is not None:
                    env_sets = self.configure_set.env_sets.split(';')
                    for env_set in env_sets:
                        f.write(f"{env_set}\n")
                f.write(f"cd {self.job_dir}\n")
                # include the resubmit script with a trap to handle unexpected exits
                f.write(f"trap 'echo 0 > {self.finished_flag_file}; echo -1 > {self.error_flag_file}' EXIT\n")
                f.write(f"{self.resubmit_script_unfinished} > {self.job_dir}/resubmit_unfinished_{self.unfinished_times}.log 2>&1\n")
                f.write(f"if [ $? -eq 0 ]; then\n")
                f.write(f"    echo 1 > {self.finished_flag_file}\n")
                f.write(f"    echo 0 > {self.error_flag_file}\n")
                f.write(f"    trap - EXIT\n")
                f.write(f"else\n")
                f.write(f"    echo 0 > {self.finished_flag_file}\n")
                f.write(f"    echo -1 > {self.error_flag_file}\n")
                f.write(f"fi\n")
            # submit the new slurm file and return the new slurm id
            result = subprocess.run(['sbatch', new_slurm_file_path], stdout=subprocess.PIPE, cwd=self.configure_set.slurm_file_path)
            new_slurm_id = result.stdout.decode('utf-8').strip().split()[-1]
            return new_slurm_id
        else:
            raise ValueError(f"No resubmit script for unfinished job {self.job_id:08x}, please check the job manually.")
        
    def update_job_status(self) -> None:
        current_slurm_id = self.current_slurm_id
        current_status = self.status
        if current_status == JobStatus.FINISHED:
            return
        # if the job is pending or running, check the status in slurm system
        if current_status in [JobStatus.PENDING, JobStatus.RUNNING]:
            slurm_status = self.check_slurm_status(str(current_slurm_id))
            if slurm_status != current_status:
                self.status = slurm_status
                if slurm_status in [JobStatus.UNFINISHED, JobStatus.FAILED]:
                    print(f"Job {self.job_id:08x} is {slurm_status} in slurm system, we updated their status in the database, run again to re-submit them.")
                self.save()
        # if the job is unfinished, re-submit and update the status accordingly
        if current_status == JobStatus.UNFINISHED:
            if self.unfinished_times >= self.max_unfinished_times: #type: ignore
                print(f"Job {self.job_id:08x} has been unfinished for {self.unfinished_times} times, which exceeds the maximum unfinished times {self.max_unfinished_times}, please check the job manually.")
                self.status = JobStatus.CHECK_REQUIRED
                self.save()
                return
            elif self.resubmit_script_unfinished is not None:
                self.unfinished_times += 1
                new_slurm_id = self.resubmit_unfinished_task()
                self.current_slurm_id = new_slurm_id
                self.status = JobStatus.PENDING
                self.save()
            else:
                print(f"Job {self.job_id:08x} is unfinished, but no resubmit script is provided, please check the job manually.")
                self.status = JobStatus.CHECK_REQUIRED
                self.save()
                return
        # if the job is failed, re-submit and update the status accordingly
        if current_status == JobStatus.FAILED:
            if self.failed_times >= self.max_failed_times: #type: ignore
                print(f"Job {self.job_id:08x} has been failed for {self.failed_times} times, which exceeds the maximum failed times {self.max_failed_times}, please check the job manually.")
                self.status = JobStatus.CHECK_REQUIRED
                self.save()
                return
            elif self.resubmit_script_failed is not None:
                self.failed_times += 1
                new_slurm_id = self.resubmit_failed_task()
                self.current_slurm_id = new_slurm_id
                self.status = JobStatus.PENDING
                self.save()
            else:
                print(f"Job {self.job_id:08x} is failed, but no resubmit script is provided, please check the job manually.")
                self.status = JobStatus.CHECK_REQUIRED
                self.save()
                return
    def check(self) -> None:
        # check this job's status, and if it is CHECK_REQUIRED, print the path of slurm output and error files and log files for manual checking.
        if self.status == JobStatus.CHECK_REQUIRED:
            print(f"Job {self.job_id:08x} is in CHECK_REQUIRED status, please check the following files for more information:")
            print(f"Slurm output file: {os.path.join(self.configure_set.slurm_file_path, f'{self.job_id:08x}_resubmit_failed_{self.failed_times}.out')}")
            print(f"Slurm error file: {os.path.join(self.configure_set.slurm_file_path, f'{self.job_id:08x}_resubmit_failed_{self.failed_times}.err')}")
            print(f"Resubmit log file (if exists): {os.path.join(str(self.job_dir), f'resubmit_failed_{self.failed_times}.log')}")
        else:
            print(f"Job {self.job_id:08x} is in {self.status} status.")

def connect_database():
    database = peewee.SqliteDatabase(os.path.expanduser(data_base_path))
    # try to connect, and check if the table exists, if not, create the table.
    database.connect()
    ConfigurationSet.bind(database)
    SlurmJob.bind(database)
    if not ConfigurationSet.table_exists():
        database.create_tables([ConfigurationSet])
    # test stage, add column to the ConfigurationSet table, the env_sets field should be added.
    # database.execute_sql("ALTER TABLE configurationset ADD COLUMN env_sets TEXT;")
    # test stage, drop the SlurmJob table.
    # database.drop_tables([SlurmJob])
    if not SlurmJob.table_exists():
        database.create_tables([SlurmJob])

def test_sql_support():
    # test whether this platform supports sqlite, and write the result to ~/.bashrc
    LastTestResult = os.environ.get("BINDGAR_SQL_SUPPORT_TEST_RESULT", None)
    if LastTestResult is not None:
        print(f"Last test result for SQL support: {LastTestResult}")
        return
    try:
        import sqlite3
        conn = sqlite3.connect(':memory:')
        conn.execute('CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)')
        conn.execute('INSERT INTO test (name) VALUES (?)', ('test',))
        result = conn.execute('SELECT * FROM test').fetchone()
        if result == (1, 'test'):
            print("echo 'This platform supports SQLite.'")
            with open(os.path.expanduser("~/.bashrc"), 'a') as f:
                f.write(f"\nexport BINDGAR_SQL_SUPPORT_TEST_RESULT='SUPPORTED'\n")
                os.environ["BINDGAR_SQL_SUPPORT_TEST_RESULT"] = "SUPPORTED"
        else:
            print("This platform does not support SQLite.")
            with open(os.path.expanduser("~/.bashrc"), 'a') as f:
                f.write(f"\nexport BINDGAR_SQL_SUPPORT_TEST_RESULT='NOT_SUPPORTED'\n")
                os.environ["BINDGAR_SQL_SUPPORT_TEST_RESULT"] = "NOT_SUPPORTED"
    except Exception as e:
        print(f"Error testing SQL support: {e}")
        print("This platform does not support SQLite.")
        with open(os.path.expanduser("~/.bashrc"), 'a') as f:
            f.write(f"\nexport BINDGAR_SQL_SUPPORT_TEST_RESULT='NOT_SUPPORTED'\n")
        os.environ["BINDGAR_SQL_SUPPORT_TEST_RESULT"] = "NOT_SUPPORTED"
def is_sql_supported() -> bool:
    # read result from environment variable
    result = os.environ.get("BINDGAR_SQL_SUPPORT_TEST_RESULT", None)
    if result is None:
        test_sql_support()
        result = os.environ.get("BINDGAR_SQL_SUPPORT_TEST_RESULT", "NOT_SUPPORTED")
    return result == "SUPPORTED"

@register_command(command_name="slurm-config", help_msg="Create a new configuration set for slurm jobs.")
def new_slurm_config():
    assert is_sql_supported(), "This platform does not support SQLite, please check the SQL support first."
    connect_database()
    import sys
    if len(sys.argv) == 1:
        config_set = ConfigurationSet.new_configuration_set()
        print(f"New configuration set created with name: {config_set.config_name}")
    elif len(sys.argv) >= 2:
        config_name = sys.argv[1]
        config_set = ConfigurationSet.get(ConfigurationSet.config_name == config_name)
        config_set.edit_configuration_set()
        print(f"Configuration set {config_set.config_name} is updated.")
    

@register_command(command_name="update-slurm-jobs", help_msg="Update the status of all slurm jobs in the database, and re-submit the unfinished or failed jobs if the resubmit script is provided.")
def update_slurm_jobs():
    assert is_sql_supported(), "This platform does not support SQLite, please check the SQL support first."
    connect_database()
    jobs = SlurmJob.select()
    for job in jobs:
        job.update_job_status()

@register_command(command_name="manage-slurm-jobs", help_msg="Check the status of all slurm jobs in the database, and edit them in vim.")
def manage_slurm_jobs():
    DEFAULT_PARAMS: InputAcceptable = {
        "configure_set_name": {
            "default": None,
            "help": "the name of the slurm configuration set to use for this job",
            "type": str,
            "short": "s",
        },
        "edit": {
            "default": False,
            "help": "whether to edit the jobs in vim, if False, show the jobs in less",
            "type": bool,
            "short": "m",
        },
    }
    assert is_sql_supported(), "This platform does not support SQLite, please check the SQL support first."
    
    input_loader = InputLoader(DEFAULT_PARAMS)
    input_params = input_loader.load()
    configure_set_name = input_params["configure_set_name"]
    edit = input_params["edit"]
    
    connect_database()
    
    if configure_set_name is not None:
        configure_set = ConfigurationSet.get(ConfigurationSet.config_name == configure_set_name)
    else:
        configure_set = ConfigurationSet.select().first()
    
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".tmp", delete=False) as tmp:
        tmp.write(b"# Please check the status of the slurm jobs in the following format:\n")
        tmp.write(b"# job_id: the id of the job in the database\n")
        tmp.write(b"# current_slurm_id: the current slurm id of the job\n")
        tmp.write(b"# job_dir: the directory where the job is running\n")
        tmp.write(b"# status: the status of the job (pending, running, finished, unfinished, failed, check_required)\n")
        tmp.write(b"# script_for_failed: the script to re-submit the job if it is failed\n")
        tmp.write(b"# script_for_unfinished: the script to re-submit the job if it is unfinished\n")
        tmp.write(b"# You can edit any information of the job, or remove them\n")
        tmp.write(b"# If you want to remove them, add a 'rm' before the job_id, for example: 'rm 1:'\n")
        
        jobs = configure_set.jobs
        for job in jobs:
            tmp.write(f"{job.job_id:08x}: {job.current_slurm_id}, {job.job_dir}, {job.status},{job.resubmit_script_failed},{job.resubmit_script_unfinished}\n".encode('utf-8'))
        
        tmp.flush()
        tmp_path = tmp.name
    
    if edit:
        old_time = os.path.getmtime(tmp_path)
        subprocess.run(['vim', tmp_path])
        new_time = os.path.getmtime(tmp_path)
        
        if new_time == old_time:
            print("No changes made to the job status, exiting.")
            os.remove(tmp_path)
            return
        
        with open(tmp_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line.startswith("#") or line == "":
                    continue
                if line.startswith("rm"):
                    job_id_str = line.split()[1].strip(':')
                    job_id = int(job_id_str, 16)
                    job = SlurmJob.get(SlurmJob.job_id == job_id)
                    job.delete_instance()
                else:
                    job_id, rest = line.split(":", 1)
                    job_id = int(job_id.strip(), 16)
                    current_slurm_id, job_dir, status, resubmit_script_failed, resubmit_script_unfinished = rest.split(",", 4)
                    current_slurm_id = current_slurm_id.strip()
                    job_dir = job_dir.strip()
                    status = status.strip()
                    resubmit_script_failed = resubmit_script_failed.strip()
                    resubmit_script_unfinished = resubmit_script_unfinished.strip()
                    
                    if configure_set.jobs.where(SlurmJob.job_id == job_id).exists():
                        job = SlurmJob.get(SlurmJob.job_id == job_id)
                        job.current_slurm_id = current_slurm_id
                        job.job_dir = job_dir
                        job.status = status
                        job.resubmit_script_failed = resubmit_script_failed
                        job.resubmit_script_unfinished = resubmit_script_unfinished
                        job.save()
                    else:
                        job = SlurmJob.create(
                            job_id=job_id,
                            current_slurm_id=current_slurm_id,
                            job_dir=job_dir,
                            status=status,
                            resubmit_script_failed=resubmit_script_failed,
                            resubmit_script_unfinished=resubmit_script_unfinished,
                            configure_set=configure_set
                        )
    else:
        subprocess.run(['less', tmp_path])
    
    os.remove(tmp_path)

#@register_command(command_name="manage-slurm-jobs", help_msg="Check the status of all slurm jobs in the database, and edit them in vim.")
def _ref_manage_slurm_jobs():
    assert is_sql_supported(), "This platform does not support SQLite, please check the SQL support first."
    # update_slurm_jobs()
    connect_database()
    if len(sys.argv) >= 2:
        configure_name = sys.argv[1]
    else:
        configure_name = None
    if configure_name is not None:
        configure_set = ConfigurationSet.get(ConfigurationSet.config_name == configure_name)
    else:
        configure_set = ConfigurationSet.select().first()
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".tmp") as tmp:
        tmp.write(b"# Please check the status of the slurm jobs in the following format:\n")
        tmp.write(b"# job_id: the id of the job in the database\n")
        tmp.write(b"# current_slurm_id: the current slurm id of the job\n")
        tmp.write(b"# job_dir: the directory where the job is running\n")
        tmp.write(b"# status: the status of the job (pending, running, finished, unfinished, failed, check_required)\n")
        tmp.write(b"# script_for_failed: the script to re-submit the job if it is failed\n")
        tmp.write(b"# script_for_unfinished: the script to re-submit the job if it is unfinished\n")
        tmp.write(b"# You can edit any information of the job, or remove them\n")
        tmp.write(b"# If you want to remove them, add a 'rm' before the job_id, for example: 'rm 1:'\n")
        jobs = configure_set.jobs
        for job in jobs:
            tmp.write(f"{job.job_id:08x}: {job.current_slurm_id}, {job.job_dir}, {job.status},{job.resubmit_script_failed},{job.resubmit_script_unfinished}\n".encode('utf-8'))
        tmp.flush()
        old_time = os.path.getmtime(tmp.name)
        subprocess.run(['vim', tmp.name])
        # 如果没有修改任何内容，直接退出。 先检查文件是否被修改过，如果没有修改过，直接返回。
        new_time = os.path.getmtime(tmp.name)
        if new_time == old_time:
            print("No changes made to the job status, exiting.")
            return
        with open(tmp.name, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line.startswith("#") or line == "":
                    continue
                if line.startswith("rm"):
                    job_id_str = line.split()[1].strip(':')
                    job_id = int(job_id_str, 16)
                    job = SlurmJob.get(SlurmJob.job_id == job_id)
                    job.delete_instance()
                else:
                    job_id, rest = line.split(":", 1)
                    job_id = int(job_id.strip(), 16)
                    current_slurm_id, job_dir, status, resubmit_script_failed, resubmit_script_unfinished = rest.split(",", 4)
                    current_slurm_id = current_slurm_id.strip()
                    job_dir = job_dir.strip()
                    status = status.strip()
                    resubmit_script_failed = resubmit_script_failed.strip()
                    resubmit_script_unfinished = resubmit_script_unfinished.strip()
                    # check if it is update or create, if the job_id exists in the database, update it, otherwise create a new job with the provided information.
                    if configure_set.jobs.where(SlurmJob.job_id == job_id).exists():
                        job = SlurmJob.get(SlurmJob.job_id == job_id)
                        job.current_slurm_id = current_slurm_id
                        job.job_dir = job_dir
                        job.status = status
                        job.resubmit_script_failed = resubmit_script_failed
                        job.resubmit_script_unfinished = resubmit_script_unfinished
                        job.save()
                    else:
                        job = SlurmJob.create(
                            job_id=job_id,
                            current_slurm_id=current_slurm_id,
                            job_dir=job_dir,
                            status=status,
                            resubmit_script_failed=resubmit_script_failed,
                            resubmit_script_unfinished=resubmit_script_unfinished,
                            configure_set=configure_set
                        )
        
@register_command(command_name="trace-slurm-job", help_msg="Register a new slurm job in the database, and print the wrappered script to submit the job with the trap to manage the finished and failed flag files.")
def trace_slurm_job():
    from ..common import print_statstic_time
    import atexit
    atexit.unregister(print_statstic_time)
    DEFAULT_PARAMS: InputAcceptable = {
    "configure_set_name": {
        "default": None,
        "help": "the name of the slurm configuration set to use for this job",
        "type": str,
        "short": "s",
    },
    "job_dir": {
        "default": None,
        "help": "the directory where the slurm job is running, and where the finished and failed flag files are stored",
        "type": str,
        "short": "p",
    },
    "current_slurm_id": {
        "default": None,
        "help": "the current slurm id of the job",
        "type": str,
        "short": "j",
    },
    "resubmit_script_failed": {
        "default": "gengaCPU -R -1 -Nomp 32",
        "help": "the script to re-submit the job if it is failed, this script should be able to run the job independently",
        "type": str,
        "short": "f",
    },
    "resubmit_script_unfinished": {
        "default": "gengaCPU -R -1 -Nomp 32",
        "help": "the script to re-submit the job if it is unfinished, this script should be able to run the job independently",
        "type": str,
        "short": "u",
    },
    "max_unfinished_times": {
        "default": 3,
        "help": "the maximum times to re-submit the job if it is unfinished, after which the job will be marked as CHECK_REQUIRED",
        "type": int,
    },
    "max_failed_times": {
        "default": 3,
        "help": "the maximum times to re-submit the job if it is failed, after which the job will be marked as CHECK_REQUIRED",
        "type": int,
    },
    "error_flag_file": {
        "default": "is_failed",
        "help": "the name of the flag file to indicate whether the job is failed, this file should be managed by the slurm task itself",
        "type": str,
    },
    "finished_flag_file": {
        "default": "is_finished",
        "help": "the name of the flag file to indicate whether the job is finished, this file should be managed by the slurm task itself",
        "type": str,
    },
    "original_command": {
        "default": None,
        "help": r"""the original command to run the job, it will be wrappered.
        For example, 
        ```
        bindgar trace-slurm-job -s defaultname \
            -p "$(pwd)" \
            -j ${SLURM_JOB_ID} \
            -f "gengaCPU -R -1 -Nomp 32" \
            -u "gengaCPU -R -1 -Nomp 32" \
            -- max_unfinished_times 5 --max_failed_times 5 \
            --error_flag_file is_failed --finished_flag_file is_finished \
            --original_command "gengaCPU -Nomp 32"
        ```
        will generate a new script like that:
        ```
        trap 'echo 0 > is_finished; echo -1 > is_failed' EXIT
        gengaCPU -Nomp 32 > job.log 2>&1
        if [ $? -eq 0 ]; then
            echo 1 > is_finished
            echo 0 > is_failed
            trap - EXIT
        else
            echo 0 > is_finished
            echo -1 > is_failed
        fi
        ```
        if you use this command by adding a `eval` + $ + (the command) in your original slurm script,
        then the generated script will be run by the bash directly,
        and the slurm job will be registered in the database with the provided information. 
        This is useful when you want to use this tool to manage your slurm jobs,
        but do not want to modify your original command too much.
        """,
        "type": str,
        "short": "o",
    },
    }
    assert is_sql_supported(), "This platform does not support SQLite, please check the SQL support first."
    connect_database()
    input_loader = InputLoader(DEFAULT_PARAMS)
    input_params = input_loader.load()
    configure_set_name = input_params["configure_set_name"]
    job_dir = input_params["job_dir"]
    current_slurm_id = input_params["current_slurm_id"]
    resubmit_script_failed = input_params["resubmit_script_failed"]
    resubmit_script_unfinished = input_params["resubmit_script_unfinished"]
    max_unfinished_times = input_params["max_unfinished_times"]
    max_failed_times = input_params["max_failed_times"]
    error_flag_file = input_params["error_flag_file"]
    finished_flag_file = input_params["finished_flag_file"]
    original_command = input_params["original_command"]
    job = SlurmJob.register_job(
        configure_set_name=configure_set_name,
        job_dir=job_dir,
        current_slurm_id=current_slurm_id,
        status=JobStatus.PENDING,
        resubmit_script_failed=resubmit_script_failed,
        resubmit_script_unfinished=resubmit_script_unfinished,
        max_unfinished_times=max_unfinished_times,
        max_failed_times=max_failed_times,
        error_flag_file=error_flag_file,
        finished_flag_file=finished_flag_file
    )
    print(f"echo \"Job {job.job_id:08x} is registered in the database with current slurm id {current_slurm_id} and status {JobStatus.PENDING}\";")
    wrapper_script = f"""trap 'echo 0 > {error_flag_file}; echo 0 > {finished_flag_file}' EXIT ; \
{original_command} > {job_dir}/job.log 2>&1; \
if [ $? -eq 0 ]; then echo 1 > {finished_flag_file}; echo 0 > {error_flag_file}; trap - EXIT; \
else echo 0 > {finished_flag_file}; echo -1 > {error_flag_file}; fi
    """
    print(wrapper_script)

if __name__ == "__main__":
    is_sql_supported()