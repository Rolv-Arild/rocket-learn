@echo off
REM sets inside of ifs annoy windows
Setlocal EnableDelayedExpansion

if exist !APPDATA!\bakkesmod\ (
    echo Bakkesmod located at !APPDATA!\bakkesmod\ 
    goto :done
    
) else (
    echo \nBakkesmod not found at !APPDATA!\bakkesmod\ 
    echo * If you've already installed it elsewhere, you're fine *
    set /p choice=Download Bakkesmod[y/n]?:
    
    if /I "!choice!" EQU "Y" goto :install
    if /I "!choice!" EQU "N" goto :no_install
)

    :install
    echo Downloading Bakkesmod
    curl.exe -L --output !USERPROFILE!\Downloads\BakkesModSetup.zip --url https://github.com/bakkesmodorg/BakkesModInjectorCpp/releases/latest/download/BakkesModSetup.zip
    tar -xf !USERPROFILE!\Downloads\BakkesModSetup.zip -C !USERPROFILE!\Downloads\
    !USERPROFILE!\Downloads\BakkesModSetup.exe
    
    if !errorlevel! neq 0 echo \n*** Problem with Bakkesmod installation. Manually install and try again ***\n
    if !errorlevel! neq 0 pause & exit /b !errorlevel!
    
    echo Bakkesmod installed!
    goto :done

    :no_install
    echo nope
    goto :done    
    
    :done

python -m venv !LocalAppData!\rocket_learn\venv

CALL  !LocalAppData!\rocket_learn\venv\Scripts\activate.bat

python -m pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

if !errorlevel! neq 0 pause & exit /b !errorlevel!

echo.
echo #########################
echo ### Launching Worker! ###
echo #########################
echo.

set "helper_name=soren"

py C:\Users\Daniel\RLGymDev\repos\rocket-learn\distributedWorker.py !helper_name!

pause