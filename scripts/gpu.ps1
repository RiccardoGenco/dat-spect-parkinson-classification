<#
.SYNOPSIS
    Esegue un comando dentro l'immagine ufficiale, con GPU e i dati montati.

.DESCRIPTION
    Il ramo a voxel gira solo nell'immagine ufficiale: il venv locale non ha
    torch. Ogni volta che serve la GPU la riga di `docker run` e' la stessa, ed
    e' lunga abbastanza da sbagliarla — e sbagliarla costa un messaggio
    fuorviante sui mount invece di un errore chiaro.

    Monta la cartella PADRE del repo su /work, non il repo: cosi' i sette
    riferimenti relativi del progetto (`../parkinson-data`) risolvono dentro il
    container esattamente come risolvono fuori, e nessun percorso assoluto entra
    nel codice.

    Da eseguire da PowerShell e non da Git Bash: Git Bash converte i percorsi in
    stile MSYS e rompe i mount con errori che non nominano il problema.

.EXAMPLE
    .\scripts\gpu.ps1 "python tests/test_metro.py"

.EXAMPLE
    .\scripts\gpu.ps1 "python scripts/16_metro_esterno.py --esterni gruppi --train-seeds 0-3" -Log metro_gruppi
#>
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Comando,

    # Nome del file sotto la cartella logs del progetto. Senza, stampa soltanto.
    [string]$Log = "",

    # La rete resta spenta per difetto: e' cosi' che gira la piattaforma, e un
    # arco che senza saperlo scaricasse dei pesi funzionerebbe qui e fallirebbe
    # in gara — nel modo peggiore, cioe' ripiegando in silenzio.
    [switch]$ConRete
)

$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
$radice = Split-Path -Parent $repo
$nomeRepo = Split-Path -Leaf $repo

$candidate = @(
    "competition-sfmn-parkinsons-runtime:gpu-local",
    "competition-sfmn-parkinsons-runtime:gpu-latest",
    "competitionsfmnparkinsonsprodacr.azurecr.io/competition-sfmn-parkinsons-runtime:gpu-latest"
)
$immagine = $null
foreach ($c in $candidate) {
    if (docker images -q $c 2>$null) { $immagine = $c; break }
}
if (-not $immagine) {
    Write-Error ("Nessuna immagine trovata fra:`n  " + ($candidate -join "`n  ") +
        "`nElenca quelle presenti con: docker images")
}

$rete = if ($ConRete) { @() } else { @("--network", "none") }

$argomenti = @(
    "run", "--rm", "--gpus", "all",
    "--platform", "linux/amd64"
) + $rete + @(
    "-v", "${radice}:/work",
    "-w", "/work/$nomeRepo",
    "-e", "PYTHONPATH=/work/$nomeRepo",
    # Senza questo, Python vede stdout su una pipe (Tee-Object) e passa al
    # buffering a blocchi: il log si ferma a meta' e sembra un container
    # bloccato. Il 3 agosto il metro con `--norm istanza` e' arrivato in fondo
    # in 80 minuti e ha scritto il suo artefatto; il log si e' fermato al fold
    # 4 su 5 e non ha mai ricevuto il blocco finale, che e' proprio la parte
    # per cui il log esterno esiste. Il guasto non perde il risultato:
    # perde la sua tracciabilita', e in silenzio.
    "-e", "PYTHONUNBUFFERED=1",
    "--entrypoint", "bash",
    # `2>&1` DENTRO il container e non fuori: PowerShell 5.1 avvolge stderr in
    # ErrorRecord e manda a capo alla larghezza della finestra, il che tronca i
    # traceback proprio quando servono. Unendo i due flussi dentro bash arrivano
    # come testo normale, interi.
    $immagine, "-lc", "$Comando 2>&1"
)

Write-Host "immagine : $immagine"
Write-Host "montaggio: $radice -> /work   (cwd /work/$nomeRepo)"
Write-Host "rete     : $(if ($ConRete) { 'attiva' } else { 'spenta' })"
Write-Host "comando  : $Comando`n"

if ($Log) {
    $cartella = Join-Path $radice "logs"
    New-Item -ItemType Directory -Force -Path $cartella | Out-Null
    $stampa = Get-Date -Format "yyyy-MM-dd_HHmm"
    $file = Join-Path $cartella "${stampa}_$Log.txt"
    # Tee-Object e non un semplice redirect: il log serve a rileggere, ma
    # guardare l'avanzamento in diretta e' cio' che fa accorgere di un arco che
    # sta sbagliando dopo due minuti invece che dopo cinquanta.
    & docker @argomenti 2>&1 | Tee-Object -FilePath $file
    Write-Host "`nlog: $file"
} else {
    & docker @argomenti 2>&1
}

exit $LASTEXITCODE
