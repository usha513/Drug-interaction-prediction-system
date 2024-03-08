/*
SQLyog Community v13.1.7 (64 bit)
MySQL - 5.5.29 : Database - myproject
*********************************************************************
*/


CREATE DATABASE /*!32312 IF NOT EXISTS*/`myproject` /*!40100 DEFAULT CHARACTER SET latin1 */;

USE `myproject`;


CREATE TABLE `users` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `username` varchar(255) NOT NULL,
  `password` varchar(255) NOT NULL,
  `email` varchar(255) NOT NULL,
  `phone` varchar(15) NOT NULL,
  `address` text,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=5 DEFAULT CHARSET=latin1;

/*Data for the table `users` */

insert  into `users`(`id`,`username`,`password`,`email`,`phone`,`address`) values 
(1,'aravind','aravind','aravind@gmail.com','9898987898','hyd');
