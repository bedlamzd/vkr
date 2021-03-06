# ВКР: "Разработка системы 3D-сканирования"

Выпускная бакалаврская работа по специальности **"Робототехника"**.

Задача приложения - сгенерировать по заданному .dxf контуру gcode для принтера с целью
отрисовки на объектах в сцене.

Этот код является частью кода из репозитория [MT.Pasticciere](https://github.com/mt-lab/MT.Pasticciere).
Цель этого репозитория в улучшении написанного кода и более удобном разграничении файлов
для оформления дипломной работы.

Код состоит из трёх частей:
* 3D-сканер

    * scanner.py
    * camera.py
    * checker.py
    
  Обрабатывает видео с камеры возвращая depthmap как изображение и облако точек

* Обработчик depthmap
    
    * cookie.py

  Принимает depthmap и облако точек (или их комбинацию) возвращая положения и ориентацию
  объектов в сцене.

* Gcode генератор

    * elements.py
    * gcoder.py

  Принимает .dxf файл, информацию об объектах и возвращает gcode инструкции для принтера

Конфиги на данный момент хранятся в json, скоро будет поддержка конфигов с помощью библиотеки configparser

Также репозиторий содержит latex исходники текста диплома, а так же скомпилированный PDF файл поснительной записки

Код написан на Python с использованием, в основном, numpy, opencv и ezdxf. Остальные пакеты нужны больше для визуализации и
 отладки.
