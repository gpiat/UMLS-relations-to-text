use umlsdb

-- FINDING RELEVANT RELATIONS

SELECT RELA, count(*) AS CNT
    FROM MRREL
    GROUP BY RELA
    ORDER BY CNT;
-- creates table with all relations and their count, sorted by count,
-- so we can easily remove relations with too few occurrences to be
-- worth the trouble.
-- Less than 12% of relations are of the first 900 relation types.
-- The inflection point of the cumsum is around 950.

SELECT CUI1, T2.STR AS STR1, RELA, T3.STR AS STR2, CUI2
    FROM (
        SELECT CUI1, RELA, CUI2
        FROM MRREL GROUP BY RELA) T1
    INNER JOIN MRCONSOABBR T2
        ON T1.CUI1=T2.CUI
    INNER JOIN MRCONSOABBR T3
        ON T1.CUI2=T3.CUI;
-- gets an example of every relation
-- saved in rela_examples.txt



-- CREATION OF TABLES WITH RELEVANT INFORMATION:
-- CUI1   |   STR1   |   RELA   |   STR2   |   CUI2
-- ONLY WITH PREFERRED STRINGS AND RELEVANT RELATIONS

-- Creating a table with only the CUIs and their preferred form
CREATE TABLE MRCONSOABBR AS
    -- SELECT CUI, TS, STT, SAB, TTY, STR, CVF, SRL
    SELECT CUI, STR
    FROM MRCONSO
    WHERE ISPREF="Y"
            -- is preferred.
            -- alone: 10M entries total
            AND STT="PF"
            -- preferred form (not related to ispref apparently)
            -- alone: 10.5M entries total
            AND TS="P"
            -- is preferred LUI (not related to STT or ISPREF apparently)
            -- alone: 6.4M entries total
        -- STT + TS + ISPREF: all distinct CUIs represented,
        --                    4.6M entries total
        -- AND SUPPRESS="N"
            -- term is not suppressed
        -- AND CVF IN (256, 2304, 4352, 6400, 8448, 10496, 16640, 33024, 49408)
            -- its content view flag has the NLP bit set
    ORDER BY CUI;
-- Query OK, 4 661 800 rows affected (33.22 sec)


-- Setting CUI as primary key as this is used for indexing and should help speed up future queries
ALTER TABLE MRCONSOABBR ADD PRIMARY KEY (CUI);
-- Query OK, 0 rows affected (13.90 sec)


-- Creating table with only CUI pairs and relations
CREATE TABLE MRRELabbr AS
    SELECT DISTINCT CUI1, RELA, CUI2
    FROM MRREL
    WHERE RELA IN ("may_treat", "may_prevent", "cause_of",
                   "causative_agent_of", "contraindicated_with_disease",
                   "isa", "associated_with", "clinically_associated_with",
                   "co-occurs_with", "has_method", "tradename_of",
                   "measures", "part_of", "member_of",
                   "finding_method_of", "possibly_equivalent_to",
                   "same_as", "active_ingredient_of",
                   "inactive_ingredient_of", "concept_in_subset",
                   "has_manifestation", "ingredient_of",
                   "classifies", "mapped_to", "consists_of",
                   "is_associated_anatomic_site_of", "occurs_in",
                   "gene_plays_role_in_process")
    AND CUI1 != CUI2;
-- Query OK, 5 928 995 rows affected (1 min 14.27 sec)

CREATE TABLE RELPREF AS
    SELECT DISTINCT CUI1, t2.STR AS STR1, RELA, t3.STR AS STR2, CUI2 
    from MRRELabbr t1
    inner join MRCONSOABBR t2
        on t1.CUI1=t2.cui
    inner join MRCONSOABBR t3
        on t1.CUI2=t3.cui;
-- Query OK, 5 928 995 rows affected (1 min 11.68 sec)

SELECT * FROM RELPREF WHERE RAND() < 0.0001;




-- TEMP
SELECT count(*)
    FROM (
        SELECT CUI1, CUI2
        FROM (
            SELECT CUI1, CUI2, RELA
            FROM MRREL
            ORDER BY CUI2) T
        WHERE RELA = "inverse_isa") T1;
-- counts number of CUIs involved in inverse-is-a relations -- I think?

CREATE TABLE SHAZ AS
    SELECT DISTINCT CUI1, RELA, CUI2
    FROM MRREL
    WHERE RELA IN ("mapped_to", "icd_dagger", "classified_as");

CREATE TABLE BOT AS
    SELECT DISTINCT CUI1, t2.STR AS STR1, RELA, t3.STR AS STR2, CUI2 
    from SHAZ t1
    inner join MRCONSOABBR t2
        on t1.CUI1=t2.cui
    inner join MRCONSOABBR t3
        on t1.CUI2=t3.cui;



SELECT T2.STR AS STR1, RELA, T3.STR AS STR2
    FROM (
        SELECT CUI1, RELA, CUI2 
        FROM MRREL
        WHERE RELA='has_method'
        AND CUI1!=CUI2
        limit 20) T1
    INNER JOIN MRCONSOABBR T2
        ON T1.CUI1=T2.CUI
    INNER JOIN MRCONSOABBR T3
        ON T1.CUI2=T3.CUI;
-- get examples of relation

