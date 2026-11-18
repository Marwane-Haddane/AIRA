; Inno Setup script for the AIRA desktop app.
; Build: powershell -ExecutionPolicy Bypass -File .\Build-Installer.ps1
; Input: dist\AIRA (from Build-Desktop.ps1)  ->  Output: installer\Output\AIRA-Setup-<version>.exe

#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif

[Setup]
AppId={{6C1E3B52-8F0D-4A7E-9C3A-2E5B7D1A4F90}
AppName=AIRA Dobot Drawing Studio
AppVersion={#AppVersion}
AppVerName=AIRA {#AppVersion}
AppPublisher=RoboGenius Robotics Lab · ENSA Fès
AppPublisherURL=https://github.com/Marwane-Haddane/AIRA
AppSupportURL=https://github.com/Marwane-Haddane/AIRA/issues
DefaultDirName={localappdata}\Programs\AIRA
DefaultGroupName=AIRA
DisableProgramGroupPage=yes
; per-user install: no admin rights needed, and the app can write its .env and archive
PrivilegesRequired=lowest
OutputDir=Output
OutputBaseFilename=AIRA-Setup-{#AppVersion}
SetupIconFile=..\docs\aira.ico
UninstallDisplayIcon={app}\AIRA.exe
Compression=lzma2/fast
SolidCompression=no
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
MinVersion=10.0

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "french"; MessagesFile: "compiler:Languages\French.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
; everything from the PyInstaller build, minus local test/user data
Source: "..\dist\AIRA\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "aira.log,.env,archive\*,.webview\*"

[Icons]
Name: "{group}\AIRA Dobot Drawing Studio"; Filename: "{app}\AIRA.exe"; IconFilename: "{app}\docs\aira.ico"
Name: "{group}\Uninstall AIRA"; Filename: "{uninstallexe}"
Name: "{autodesktop}\AIRA"; Filename: "{app}\AIRA.exe"; IconFilename: "{app}\docs\aira.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\AIRA.exe"; Description: "{cm:LaunchProgram,AIRA}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; app caches only; your .env key and archive/ drawings are kept
Type: filesandordirs; Name: "{app}\.webview"
Type: files; Name: "{app}\aira.log"
