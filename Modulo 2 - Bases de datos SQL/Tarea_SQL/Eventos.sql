-- PABLO PEREZ CALVO
DROP DATABASE IF EXISTS Organizaeventos;
CREATE DATABASE Organizaeventos;
USE Organizaeventos;

DROP TABLE IF EXISTS Evento;
DROP TABLE IF EXISTS Ubicacion;
DROP TABLE IF EXISTS Actividad;
DROP TABLE IF EXISTS Asistente;
DROP TABLE IF EXISTS Artista;
DROP TABLE IF EXISTS Participa; 
DROP TABLE IF EXISTS Asiste;

-- CREACIÓN DE TABLAS

CREATE TABLE Actividad (
    IdActividad SMALLINT PRIMARY KEY,
    NomActividad VARCHAR(100),
    Coste NUMERIC(10,2) DEFAULT 0,
    Tipo VARCHAR(50)
);

CREATE TABLE Ubicacion (
    IdUbicacion SMALLINT PRIMARY KEY,
    NomUbicacion VARCHAR(50),
    PrecioAlquiler NUMERIC(10,2) NOT NULL,
    Direccion VARCHAR(100) NOT NULL,
    Aforo NUMERIC(10,0),
    Localidad VARCHAR(20),
    Caracteristicas VARCHAR(200)
);

CREATE TABLE Evento (
    IdEvento SMALLINT PRIMARY KEY,
    NomEvento VARCHAR(100) NOT NULL,
    Fecha DATE NOT NULL,
    Hora TIME,
    PrecioEvento NUMERIC(6,2) NOT NULL,
    Descripcion VARCHAR(500),
    CodActividad SMALLINT,
    CodUbicacion SMALLINT,
    FOREIGN KEY (CodActividad) REFERENCES Actividad(IdActividad)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    FOREIGN KEY (CodUbicacion) REFERENCES Ubicacion(IdUbicacion)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);

CREATE TABLE Asistente(
    IdAsistente SMALLINT PRIMARY KEY,
    NomAsistente VARCHAR(50) NOT NULL,
    Email VARCHAR(50),
    Telefono Numeric(9,0)
);

CREATE TABLE Artista(
    IdArtista SMALLINT PRIMARY KEY,
    NomArtista VARCHAR(50) NOT NULL,
    Biografia VARCHAR(500)
);

CREATE TABLE Participa (
    CodArtista SMALLINT,
    CodActividad SMALLINT,
    CacheA NUMERIC(10,2),
    PRIMARY KEY (CodArtista, CodActividad),
    FOREIGN KEY (CodArtista) REFERENCES Artista(IdArtista) ON DELETE CASCADE,
    FOREIGN KEY (CodActividad) REFERENCES Actividad(IdActividad) ON DELETE CASCADE
);

CREATE TABLE Asiste (
    CodAsistente SMALLINT,
    CodEvento SMALLINT,
    PRIMARY KEY (CodAsistente, CodEvento),
    FOREIGN KEY (CodAsistente) REFERENCES Asistente(IdAsistente) ON DELETE CASCADE,
    FOREIGN KEY (CodEvento) REFERENCES Evento(IdEvento) ON DELETE CASCADE,
    Calificacion NUMERIC(2,0)
);

-- TRIGGERS

DELIMITER //
-- 1. TRIGGER para actualizar el coste de una actividad despues de añadir un artista con su respectivo cache
CREATE TRIGGER AñadeCosteActividad
AFTER INSERT ON Participa
FOR EACH ROW
BEGIN
    UPDATE Actividad
    SET Coste = Coste + NEW.CacheA
    WHERE IdActividad = NEW.CodActividad;
END;
//
-- 2. TRIGGER para actualizar el coste de una actividad tras eliminar un artista de dicha actividad
CREATE TRIGGER RestaCosteActividad
AFTER DELETE ON Participa
FOR EACH ROW
BEGIN
    UPDATE Actividad
    SET Coste = Coste - OLD.CacheA
    WHERE IdActividad = OLD.CodActividad;
END;
//

-- 3. TRIGGER para verificar que el aforo maximo de cada ubicación no se supera al añadir asistentes
CREATE TRIGGER VerificarAforo
BEFORE INSERT ON Asiste
FOR EACH ROW
BEGIN
    DECLARE aforomax INT;
    DECLARE numasistentes INT;
    SELECT ubi.Aforo INTO aforomax
    FROM Evento as ev
    JOIN Ubicacion as ubi ON ev.CodUbicacion = ubi.IdUbicacion
    WHERE ev.IdEvento = NEW.CodEvento;
  
    SELECT COUNT(*) INTO numasistentes
    FROM Asiste
    WHERE CodEvento = NEW.CodEvento;
    
    IF numasistentes >= aforomax THEN
        SIGNAL SQLSTATE '45523' SET MESSAGE_TEXT = 'El aforo esta completo, no quedan más entradas disponibles para este evento';
    END IF;
END;
DELIMITER ;

-- INSERCION DE DATOS

LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/Actividades.csv'  
INTO TABLE actividad
CHARACTER SET latin1  
FIELDS TERMINATED BY ';'  
LINES TERMINATED BY '\n'
IGNORE 1 rows  
(IdActividad, NomActividad, Tipo );

LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/Ubicacion.csv'  
INTO TABLE ubicacion
CHARACTER SET latin1  
FIELDS TERMINATED BY ';'  
LINES TERMINATED BY '\n'
IGNORE 1 rows  
(IdUbicacion, NomUbicacion, PrecioAlquiler, Direccion, Aforo, Localidad, Caracteristicas);

LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/Evento.csv'  
INTO TABLE evento
CHARACTER SET latin1  
FIELDS TERMINATED BY ';'  
LINES TERMINATED BY '\n'
IGNORE 1 rows  
(IdEvento, NomEvento, Fecha, Hora, PrecioEvento, Descripcion, CodActividad, CodUbicacion);

LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/Asistentes.csv'  
INTO TABLE asistente
CHARACTER SET latin1  
FIELDS TERMINATED BY ';'  
LINES TERMINATED BY '\n'
IGNORE 1 rows  
(IdAsistente, NomAsistente, Email, Telefono);

LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/Asiste.csv'  
INTO TABLE Asiste
CHARACTER SET latin1  
FIELDS TERMINATED BY ';'  
LINES TERMINATED BY '\n'
IGNORE 1 rows  
(CodAsistente, CodEvento, Calificacion);

LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/Artistas.csv'  
INTO TABLE Artista
CHARACTER SET latin1  
FIELDS TERMINATED BY ';'  
LINES TERMINATED BY '\n'
IGNORE 1 rows  
(IdArtista, NomArtista, Biografia);

LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/Participa.csv'  
INTO TABLE participa
CHARACTER SET latin1  
FIELDS TERMINATED BY ';'  
LINES TERMINATED BY '\n'
IGNORE 1 rows  
(CodArtista, CodActividad, CacheA);


-- CONSULTAS

-- 4. Obtener el nombre de los asistentes que han ido a la Final del Concurso de Agrupaciones Carnavalescas
SELECT NomAsistente
FROM asistente, asiste
WHERE asistente.IdAsistente = asiste.CodAsistente AND asiste.CodEvento = 1;

-- 5. Obtener todos los nombres de Eventos que se realizan en la Provincia de Cadiz (ya sea un pueblo o Cádiz capital)
SELECT ev.NomEvento, ubi.NomUbicacion, ubi.Localidad
FROM evento as ev, ubicacion as ubi
WHERE ubi.Localidad LIKE '%Cadiz%' and ubi.IdUbicacion = ev.CodUbicacion
order by ev.NomEvento ;

-- 6. Obtener un ranking ordenado de los eventos por su calificación media dada por los asistentes
-- y el nombre de la persona que mejor nota haya calificado al evento.
SELECT ev.NomEvento, AVG(asis.Calificacion) AS Calificacionmedia,
	(SELECT a.NomAsistente FROM Asistente as a
    JOIN Asiste AS asis ON a.IdAsistente = asis.CodAsistente
    WHERE asis.CodEvento = ev.IdEvento
    ORDER BY asis.Calificacion DESC, a.NomAsistente ASC
     LIMIT 1) AS MejorAsistente
FROM Evento as ev
JOIN Asiste as asis ON ev.IdEvento = asis.CodEvento
GROUP BY ev.IdEvento, ev.NomEvento
ORDER BY Calificacionmedia DESC;

-- 7. Seleccionar las actividades con los artistas más destacados (con un cache mayor a 5000 euros)
SELECT ac.NomActividad, ar.NomArtista, par.CacheA
FROM Participa as par
JOIN Actividad as ac ON par.CodActividad = ac.IdActividad
JOIN Artista as ar ON par.CodArtista = ar.IdArtista
WHERE par.CacheA > 5000;

-- 8. Queremos conocer todo el detalle de los eventos que se realizarán desde la fecha actual hasta dentro de 30 días
SELECT ev.NomEvento, ev.Fecha, ev.Hora, ac.NomActividad , ubi.NomUbicacion, ubi.Localidad,
    GROUP_CONCAT(DISTINCT ar.NomArtista SEPARATOR ', ') AS Artistas,
    COUNT(DISTINCT asis.CodAsistente) AS TotalAsistentes, ev.PrecioEvento
FROM Evento as ev
LEFT JOIN Actividad as ac ON ev.CodActividad = ac.IdActividad
LEFT JOIN Ubicacion as ubi ON ev.CodUbicacion = ubi.IdUbicacion
LEFT JOIN Participa as par ON ac.IdActividad = par.CodActividad
LEFT JOIN Artista  as ar ON par.CodArtista = ar.IdArtista
LEFT JOIN Asiste as asis ON ev.IdEvento = asis.CodEvento
WHERE ev.Fecha BETWEEN CURDATE() AND DATE_ADD(CURDATE(), INTERVAL 30 DAY)
GROUP BY ev.IdEvento 
ORDER BY ev.Fecha ASC;

-- VISTAS Y CONSULTAS UTILIZANDO LAS VISTAS

-- Crear una vista que calcula los ingresos generados por cada evento
CREATE VIEW IngresosEvento AS 
SELECT ev.IdEvento, ev.NomEvento, COUNT(asis.CodAsistente) * ev.PrecioEvento AS IngresosTotales
FROM Evento as ev
JOIN Asiste as asis ON ev.IdEvento = asis.CodEvento
GROUP BY ev.IdEvento;

-- 9. Seleccionar aquellos eventos cuyos ingresos totales hayan sido más de 100 euros
SELECT * FROM IngresosEvento WHERE IngresosTotales > 100;

-- 10. Determinar los eventos que generan pérdidas (Ingresos actuales < Inversion en el evento)
SELECT inev.NomEvento, ubi.PrecioAlquiler , inev.IngresosTotales,
    (inev.IngresosTotales - ubi.PrecioAlquiler - ac.Coste) AS Beneficio
FROM IngresosEvento as inev
JOIN  Evento as ev ON inev.IdEvento = ev.IdEvento
JOIN Actividad as ac ON ac.IdActividad = ev.CodActividad
JOIN  Ubicacion as ubi ON ev.CodUbicacion = ubi.IdUbicacion
WHERE (inev.IngresosTotales - ubi.PrecioAlquiler - ac.Coste) < 0
ORDER BY Beneficio ASC;

-- Crear una vista que permita saber cuantos asistentes va a cada evento
CREATE VIEW AsistentesPorEvento AS
SELECT ev.IdEvento, ev.NomEvento, COUNT(DISTINCT asis.CodAsistente) AS NumeroAsistentes
FROM Evento as ev
JOIN Asiste as asis ON ev.IdEvento = asis.CodEvento
GROUP BY ev.IdEvento;

-- 11. Obtener el Nombre de los eventos con baja ocupación, es decir, no superen un 60% del porcentaje de llenado
SELECT ev.NomEvento, ubi.NomUbicacion, ubi.Aforo, asiev.Numeroasistentes,
       CONCAT(ROUND( asiev.Numeroasistentes / ubi.Aforo * 100, 2), '%') AS PorcentajeAsistencia
FROM AsistentesporEvento as asiev
JOIN Evento as ev ON asiev.IdEvento = ev.IdEvento
JOIN Ubicacion as ubi ON ev.CodUbicacion = ubi.IdUbicacion
JOIN Asiste as asis ON ev.IdEvento = asis.CodEvento
GROUP BY ev.IdEvento
HAVING PorcentajeAsistencia < 60
ORDER BY PorcentajeAsistencia ASC;

-- 12. Determinar el nombre de los eventos a los que asistan más personas que al evento "Big Data Conference Europe 2024"
SELECT ev.NomEvento, asiev.NumeroAsistentes
FROM AsistentesporEvento as asiev
JOIN Evento as ev ON asiev.IdEvento = ev.IdEvento
WHERE asiev.NumeroAsistentes > (
    SELECT asiev.NumeroAsistentes
    FROM AsistentesporEvento as asiev
    JOIN Evento as ev ON asiev.IdEvento = ev.IdEvento
    WHERE ev.NomEvento = 'Big Data Conference Europe 2024'
);

-- COMPROBACION DE TRIGGERS
INSERT INTO Participa(CodArtista,CodActividad,CacheA) VALUES (2,1,666);

DELETE FROM Participa WHERE CodArtista = 1 AND CodActividad = 1;