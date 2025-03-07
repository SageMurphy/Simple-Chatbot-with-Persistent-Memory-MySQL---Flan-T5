create database chatbot_db;

CREATE USER 'chatbot_user'@'localhost' IDENTIFIED BY '12345';

grant all privileges on	chatbot_db.*to 'chatbot_user'@'localhost';
flush privileges;

use chatbot_db;
show  tables from chatbot_db;
select * from conversations;