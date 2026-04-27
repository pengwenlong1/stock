/*
 Navicat Premium Dump SQL

 Source Server         : aliyun
 Source Server Type    : MySQL
 Source Server Version : 80046 (8.0.46)
 Source Host           : 101.133.164.138:3306
 Source Schema         : stock

 Target Server Type    : MySQL
 Target Server Version : 80046 (8.0.46)
 File Encoding         : 65001

 Date: 26/04/2026 13:20:06
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for stock_daily_metrics
-- ----------------------------
DROP TABLE IF EXISTS `stock_daily_metrics`;
CREATE TABLE `stock_daily_metrics`  (
  `id` bigint UNSIGNED NOT NULL AUTO_INCREMENT,
  `stock_id` varchar(10) UNSIGNED NOT NULL COMMENT '关联 stock_info.stock_id',
  `trade_date` date NOT NULL COMMENT '交易日（工作日）',
  `close_price` decimal(10, 2) NOT NULL COMMENT '收盘价（代表当日股价）',
  `ma5` decimal(10, 2) NULL DEFAULT NULL COMMENT '5日均线',
  `ma10` decimal(10, 2) NULL DEFAULT NULL COMMENT '10日均线',
  `ma5_ma10_dead_cross` tinyint(1) NOT NULL DEFAULT 0 COMMENT '5日与10日均线死叉：0-否, 1-是',
  `rsi_daily` decimal(5, 2) NULL DEFAULT NULL COMMENT '日线RSI（0~100）',
  `rsi_weekly` decimal(5, 2) NULL DEFAULT NULL COMMENT '周线RSI',
  `macd` decimal(10, 4) NULL DEFAULT NULL COMMENT 'MACD值',
  `sar` decimal(10, 4) NULL DEFAULT NULL COMMENT 'sar值',
  `is_daily_top_divergence` tinyint(1) NOT NULL DEFAULT 0 COMMENT '是否日线顶背离：0-否, 1-是',
  `is_daily_bottom_divergergence` tinyint(1) NOT NULL DEFAULT 0 COMMENT '是否热线底背离：0-否, 1-是',
  `is_weekly_top_divergence` tinyint(1) NOT NULL DEFAULT 0 COMMENT '是否周线顶背离：0-否, 1-是',
  `is_weekly_bottom_divergergence` tinyint(1) NOT NULL DEFAULT 0 COMMENT '是否周线底背离：0-否, 1-是',
  `is_sar_dead_cross` tinyint(1) NOT NULL DEFAULT 0 COMMENT 'SAR死叉：0-否, 1-是',
  `is_local_high` tinyint(1) NOT NULL DEFAULT 0 COMMENT '是否局部高点：0-否, 1-是',
  `is_local_low` tinyint(1) NOT NULL DEFAULT 0 COMMENT '是否局部低点：0-否, 1-是',
  `status` tinyint(1) NOT NULL DEFAULT 0 COMMENT '计算状态：0-完成, 1-有遗漏',
  `created_at` datetime NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` datetime NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `uk_stock_date`(`stock_id` ASC, `trade_date` ASC) USING BTREE,
  INDEX `idx_trade_date`(`trade_date` ASC) USING BTREE,
  INDEX `idx_stock_status`(`stock_id` ASC, `status` ASC) USING BTREE,
  INDEX `idx_date_status`(`trade_date` ASC, `status` ASC) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 215041 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Table structure for stock_index_daily_metrics
-- ----------------------------
DROP TABLE IF EXISTS `stock_index_daily_metrics`;
CREATE TABLE `stock_index_daily_metrics`  (
  `id` bigint UNSIGNED NOT NULL AUTO_INCREMENT,
  `stock_id` varchar(10) UNSIGNED NOT NULL COMMENT '指数的code id',
  `trade_date` date NOT NULL COMMENT '交易日（工作日）',
  `close_price` decimal(10, 2) NOT NULL COMMENT '收盘价（代表当日股价）',
  `ma5` decimal(10, 2) NULL DEFAULT NULL COMMENT '5日均线',
  `ma10` decimal(10, 2) NULL DEFAULT NULL COMMENT '10日均线',
  `ma5_ma10_dead_cross` tinyint(1) NOT NULL DEFAULT 0 COMMENT '5日与10日均线死叉：0-否, 1-是',
  `rsi_daily` decimal(5, 2) NULL DEFAULT NULL COMMENT '日线RSI（0~100）',
  `rsi_weekly` decimal(5, 2) NULL DEFAULT NULL COMMENT '周线RSI',
  `macd` decimal(10, 4) NULL DEFAULT NULL COMMENT 'MACD值',
  `sar` decimal(10, 4) NULL DEFAULT NULL COMMENT 'sar值',
  `sar_line` decimal(10, 4) NULL DEFAULT NULL COMMENT 'sar金叉死叉的临界值',
  `is_daily_top_divergence` tinyint(1) NOT NULL DEFAULT 0 COMMENT '是否日线顶背离：0-否, 1-是',
  `is_daily_bottom_divergergence` tinyint(1) NOT NULL DEFAULT 0 COMMENT '是否热线底背离：0-否, 1-是',
  `is_weekly_top_divergence` tinyint(1) NOT NULL DEFAULT 0 COMMENT '是否周线顶背离：0-否, 1-是',
  `is_weekly_bottom_divergergence` tinyint(1) NOT NULL DEFAULT 0 COMMENT '是否周线底背离：0-否, 1-是',
  `is_sar_dead_cross` tinyint(1) NOT NULL DEFAULT 0 COMMENT 'SAR死叉：0-否, 1-是',
  `is_local_high` tinyint(1) NOT NULL DEFAULT 0 COMMENT '是否局部高点：0-否, 1-是',
  `is_local_low` tinyint(1) NOT NULL DEFAULT 0 COMMENT '是否局部低点：0-否, 1-是',
  `status` tinyint(1) NOT NULL DEFAULT 0 COMMENT '计算状态：0-完成, 1-有遗漏',
  `created_at` datetime NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` datetime NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `uk_stock_date`(`stock_id` ASC, `trade_date` ASC) USING BTREE,
  INDEX `idx_trade_date`(`trade_date` ASC) USING BTREE,
  INDEX `idx_stock_status`(`stock_id` ASC, `status` ASC) USING BTREE,
  INDEX `idx_date_status`(`trade_date` ASC, `status` ASC) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '用来记录大盘指数的信息' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Table structure for stock_info
-- ----------------------------
DROP TABLE IF EXISTS `stock_info`;
CREATE TABLE `stock_info`  (
  `stock_id` varchar(10) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '股票代码（600000）',
  `stock_name` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '股票名称',
  `listing_date` date NOT NULL COMMENT '上市日期',
  `created_at` datetime NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` datetime NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  UNIQUE INDEX `stock_id`(`stock_id` ASC) USING BTREE,
  UNIQUE INDEX `uk_stock_id`(`stock_id` ASC) USING BTREE,
  INDEX `idx_stock_code`(`stock_id` ASC) USING BTREE,
  INDEX `idx_listing_date`(`listing_date` ASC) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Table structure for trading_calendar
-- ----------------------------
DROP TABLE IF EXISTS `trading_calendar`;
CREATE TABLE `trading_calendar`  (
  `trade_date` date NOT NULL COMMENT '工作日（交易日）',
  PRIMARY KEY (`trade_date`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci ROW_FORMAT = Dynamic;

SET FOREIGN_KEY_CHECKS = 1;
