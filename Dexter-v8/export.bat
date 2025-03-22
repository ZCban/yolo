@echo off
REM Script Batch per eseguire il comando export.py
REM Configurazione ambiente

echo Esecuzione del comando Python per esportare il modello...

python export2.py -o best.onnx -e best32.trt --end2end --v8 

REM Controllo del risultato
IF %ERRORLEVEL% EQU 0 (
    echo Comando eseguito con successo.
) ELSE (
    echo Si è verificato un errore durante l'esecuzione del comando.
    pause
)

pause
