### Install and start postgresql

```
## Install
sudo apt update
sudo apt install postgresql postgresql-contrib -y

## Start
sudo systemctl start postgresql
sudo systemctl enable postgresql

## Check running status
sudo systemctl status postgresql # (Would be "active" if it is running properly)
```

### Login and make user

```
sudo -u postgres psql -c "ALTER USER postgres WITH PASSWORD '<your_password>';"
# Enter the password inside single quotes in '<your_password>'.
# This will make a user with user "postgres" and whatever you provided as password. 
# Is necessary in setup later.
```

### Add the password to env
```
POSTGRES_DC_PASSWORD=<your_password>
```

### Run setup file
```
python data_collection_db/setup_collection_db.py
```

### Run miners and validators as usual
