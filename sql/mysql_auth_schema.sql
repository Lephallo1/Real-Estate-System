CREATE DATABASE IF NOT EXISTS lesotho_property_ai_app;
USE lesotho_property_ai_app;

CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(80) NOT NULL UNIQUE,
    email VARCHAR(160) NOT NULL UNIQUE,
    full_name VARCHAR(120) NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role ENUM('admin', 'customer') NOT NULL,
    address VARCHAR(255) NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS login_audit (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NULL,
    username_attempt VARCHAR(80) NOT NULL,
    login_status ENUM('success', 'failure') NOT NULL,
    role_at_login ENUM('admin', 'customer') NULL,
    ip_address VARCHAR(64) NULL,
    user_agent VARCHAR(255) NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_login_user
        FOREIGN KEY (user_id) REFERENCES users(id)
        ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS customer_search_requests (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    listing_intent ENUM('sale', 'rent') NOT NULL DEFAULT 'sale',
    budget_min DECIMAL(12,2) NULL,
    budget_max DECIMAL(12,2) NULL,
    preferred_districts JSON NULL,
    preferred_bedrooms INT NULL,
    preferred_language ENUM('en', 'st') NOT NULL DEFAULT 'en',
    free_text_preference_en TEXT NULL,
    free_text_preference_st TEXT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_search_user
        FOREIGN KEY (user_id) REFERENCES users(id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS recommendation_runs (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    search_request_id BIGINT NULL,
    top_n INT NOT NULL DEFAULT 3,
    listing_intent ENUM('sale', 'rent') NOT NULL DEFAULT 'sale',
    properties_considered INT NOT NULL DEFAULT 0,
    matches_generated INT NOT NULL DEFAULT 0,
    mean_top_match_score DECIMAL(6,4) NULL,
    artifact_prefix VARCHAR(80) NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_recommendation_user
        FOREIGN KEY (user_id) REFERENCES users(id)
        ON DELETE CASCADE,
    CONSTRAINT fk_recommendation_search
        FOREIGN KEY (search_request_id) REFERENCES customer_search_requests(id)
        ON DELETE SET NULL
);
