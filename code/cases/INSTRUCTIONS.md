# Case YAML Instructions

Questo file documenta tutte le variabili YAML lette da
`cases/run_case.py`. Al momento il runner supporta i casi
`falling_square` e `falling_cylinder`.

Comando base:

```bash
python3 cases/run_case.py cases/configs/falling_square.yaml
```

Controllo senza simulare:

```bash
python3 cases/run_case.py cases/configs/falling_square.yaml --dry-run
```

## Note Generali

- I path relativi in `output.folder` sono interpretati rispetto alla root del
  progetto.
- Se `PyYAML` e' installato viene usato quello. Altrimenti il runner usa un
  parser minimale che supporta mapping annidati, scalari e liste inline tipo
  `[0.0, -9.81]`.
- Il solido e' sempre assemblato come mesh quadrilatera Q1.
- Per non avere Dirichlet sulla struttura, usa flag di bordo diversi da `-1`.
  Il default del caso e' `boundary_flag: -2`.

## Struttura Completa

```yaml
case:
  name: falling_square
  kind: falling_square
  method: id

time:
  dt: 0.001
  Nt: 20
  history_stride: 1

fluid:
  mode: free_fall
  lx: 1.0
  ly: 1.0
  resolution: 0.05
  rho: 1.0
  mu: 0.01

solid:
  size: 0.2
  center: [0.5, 0.5]
  nx: 4
  ny: 4
  rho: 1.2
  boundary_flag: -2
  material:
    E: 1000.0
    nu: 0.3
  body_force:
    acceleration: [0.0, -9.81]
    density_scale: buoyant

coupling:
  transfer_kind: ID

output:
  folder: cases/outputs/falling_square
```

## `case`

`name`
: Stringa. Nome logico del caso. Default: nome del file YAML senza estensione.
  Usato anche come default per `output.file_base`.

`kind`
: Stringa. Valori ammessi: `falling_square`, `falling_cylinder`.
  Default: `falling_square`.
  Serve a scegliere il costruttore del caso.

`method`
: Stringa. Valori ammessi: `id`, `ib`. Default: `id`.
  `id` usa una formulazione immersed-domain. `ib` usa una formulazione
  immersed-boundary; in questo caso `transfer_kind` viene di fatto fissato a
  `IB`.

## `time`

`dt`
: Float positivo. Default: `0.001`.
  Passo temporale della simulazione.

`Nt`
: Intero non negativo. Default: `10`.
  Numero di step temporali.

`steps`
: Intero non negativo. Alternativa a `Nt`; usato solo se `Nt` non e' presente.

`history_stride`
: Intero maggiore o uguale a `1`. Default: `1`.
  Salva un frame ogni `history_stride` step, piu' sempre l'ultimo frame.
  Puo' anche essere messo in `output.history_stride`; il valore in `time`
  ha precedenza.

## `fluid`

`mode`
: Stringa. Valori ammessi: `free_fall`, `ff`, `closed`, `channel`, `cf`.
  Default: `free_fall`.
  `free_fall` usa condizioni al bordo chiuse da benchmark di caduta libera.
  `channel` usa il solver canale standard con inlet/outlet.

`lx`
: Float positivo. Default: `1.0`.
  Lunghezza del dominio fluido in direzione x.

`length`
: Float positivo. Alias di `lx`; usato solo se `lx` non e' presente.

`ly`
: Float positivo. Default: `1.0`.
  Altezza del dominio fluido in direzione y.

`height`
: Float positivo. Alias di `ly`; usato solo se `ly` non e' presente.

`resolution`
: Float positivo. Se presente, determina la griglia con
  `nx = round(lx / resolution)` e `ny = round(ly / resolution)`.

`nx`
: Intero positivo. Default: `20`.
  Numero di celle fluide in x. Usato solo se `resolution` non e' presente.

`ny`
: Intero positivo. Default: `20`.
  Numero di celle fluide in y. Usato solo se `resolution` non e' presente.

`rho`
: Float positivo. Default: `1.0`.
  Densita' del fluido.

`mu`
: Float non negativo. Default: `0.01`.
  Viscosita' dinamica. Il runner passa al solver `nu = mu / rho`.

`G`
: Float. Default: `0.0`.
  Termine usato dall'assemblaggio pressione del canale. In `free_fall` deve
  restare `0.0`, perche' `fluid_assembly_ff` rifiuta valori non nulli.

`pressure_solver`
: Stringa. Default: `superlu`.
  Per `channel`: `superlu`, `splu`, `lu`, `sparse`, oppure `dct`, `fft`,
  `fast`. In `free_fall` il runner usa `fluid_assembly_ff`, quindi questa
  opzione non cambia il solver.

`inlet_velocity`
: Float. Default: `0.0`.
  Velocita' orizzontale uniforme all'ingresso per il solver `channel`.

`vel0`
: Float. Alias di `inlet_velocity`; usato solo se `inlet_velocity` non e'
  presente.

`initialize`
: Booleano. Default: `false`.
  Se `true`, inizializza il campo fluido dal profilo `inlet_velocity`.

`periodic_x`
: Booleano. Default: `false`.
  Solo per `free_fall`. Attiva periodicita' in x.

`zero_mean_periodic_x`
: Booleano. Default: `false`.
  Solo per `free_fall` con `periodic_x: true`. Rimuove la velocita' media in x.

`hesch_benchmark`
: Booleano. Default: `false`.
  Solo per `free_fall`. Usa la variante di condizioni al bordo Hesch.
  Non puo' essere combinato con `periodic_x: true`.

`body_acceleration`
: Lista inline di due float, per esempio `[0.0, -9.81]`.
  Solo per `free_fall`. Accelerazione di corpo applicata al fluido.

`assembly_kwargs`
: Mapping. Default: `{}`.
  Argomenti extra passati alla funzione di assemblaggio fluido. In `free_fall`
  il runner inserisce automaticamente `periodic_x` e `hesch_benchmark` se non
  sono gia' presenti.

`step_kwargs`
: Mapping. Default: `{}`.
  Argomenti extra passati al passo fluido. In `free_fall` il runner inserisce
  automaticamente `periodic_x`, `zero_mean_periodic_x`, `hesch_benchmark` e,
  se presente, `fluid_body_acceleration`.

## `solid`

`size`
: Float positivo. Default: `0.2`.
  Lato del quadrato. Usato come default sia per `lx` sia per `ly`.

`lx`
: Float positivo. Default: `width`, poi `size`, poi `0.2`.
  Larghezza del solido.

`width`
: Float positivo. Alias di `lx`; usato solo se `lx` non e' presente.

`ly`
: Float positivo. Default: `height`, poi `size`, poi `lx`.
  Altezza del solido.

`height`
: Float positivo. Alias di `ly`; usato solo se `ly` non e' presente.

`center`
: Lista inline di due float. Default: centro del dominio fluido.
  Posizione iniziale del centro del rettangolo: `[x, y]`.

`resolution`
: Float positivo. Se presente per `falling_square`, determina la mesh solida
  con `nx = round(lx / resolution)` e `ny = round(ly / resolution)`.
  Ha precedenza su `nx`, `ny`, `elements_x` e `elements_y`.

`nx`
: Intero positivo. Default: `elements_x`, poi `4`. Usato solo se
  `resolution` non e' presente.
  Numero di elementi Q1 in x.

`ny`
: Intero positivo. Default: `elements_y`, poi `nx`. Usato solo se
  `resolution` non e' presente.
  Numero di elementi Q1 in y.

`elements_x`
: Intero positivo. Alias di `nx`; usato solo se `resolution` e `nx` non sono
  presenti.

`elements_y`
: Intero positivo. Alias di `ny`; usato solo se `resolution` e `ny` non sono
  presenti.

`rho`
: Float. Default: `density`, poi `1.2`.
  Densita' del solido.

`density`
: Float. Alias di `rho`; usato solo se `rho` non e' presente.

`boundary_flag`
: Intero. Default: `-2`.
  Flag applicato a tutti i lati se `boundary_flags` non e' presente.
  Nota importante: il flag `-1` viene interpretato dal solver come Dirichlet,
  quindi fissa i nodi di bordo. Per una struttura libera usare `-2`.

`boundary_flags`
: Lista inline di quattro interi in ordine `[bottom, right, top, left]`.
  Se presente sostituisce `boundary_flag` lato per lato.

Per `falling_cylinder`, al posto di `size`, `lx`, `ly`, `nx` e `ny`, usa:

`radius`
: Float positivo. Raggio del cilindro. Default: `cylinder_radius`, poi meta'
  di `diameter`/`cylinder_diameter`, poi `0.125`.

`diameter`
: Float positivo. Diametro del cilindro. Usato solo per ricavare il raggio
  quando `radius`/`cylinder_radius` non sono presenti.

`resolution`
: Float positivo. Lunghezza target dei lati nella mesh triangolare P1 del
  cilindro.

`edge_length`
: Float positivo. Alias storico di `resolution` per `falling_cylinder`.
  Default: `target_edge_length`, poi `resolution`, poi la minima spaziatura
  fluida.

### `solid.material`

Puoi specificare il materiale in due modi.

`E`
: Float positivo. Default: `1000.0`.
  Modulo di Young. Usato insieme a `nu` se non dai direttamente `lambda` e
  `mu`.

`nu`
: Float in `(-1, 0.5)`. Default: `0.3`.
  Coefficiente di Poisson.

`plane_stress`
: Booleano. Default: `false`.
  Se `true`, calcola `lambda` con ipotesi di sforzo piano.

`lambda`
: Float. Parametro di Lame. Se presente insieme a `mu`, il runner usa questi
  valori e ignora `E`, `nu`, `plane_stress`.

`mu`
: Float. Secondo parametro di Lame. Deve essere presente insieme a `lambda`.

### `solid.body_force`

`body_force` puo' essere:

- assente o `null`: forza nulla;
- lista inline `[fx, fy]`: forza volumetrica costante gia' scalata;
- mapping con `force_density`;
- mapping con `acceleration` e `density_scale`.

`force_density`
: Lista inline di due float `[fx, fy]`.
  Forza volumetrica costante applicata direttamente al solido.

`acceleration`
: Lista inline di due float. Default: `[0.0, 0.0]`.
  Accelerazione da trasformare in forza volumetrica tramite `density_scale`.

`density_scale`
: Stringa o numero. Default: `buoyant`.
  Valori ammessi:
  `buoyant`, `difference`, `rho_s-rho_f`, `rho_s_minus_rho_f`
  usano `rho_s - rho_f`;
  `solid`, `rho_s` usano `rho_s`;
  `fluid`, `rho_f` usano `rho_f`;
  `unit`, `none` usano `1.0`;
  un numero in stringa, per esempio `"2.5"`, usa quel valore.

## `coupling`

`transfer_kind`
: Stringa. Default: `ID`.
  Per `method: id`: `ID`, `MIX`, `MIX_noC`.
  Alias accettati dal solver includono `id`, `mix`, `mixed`, `mix_noc`,
  `mix_no_c`, `mix-noc`. Per `method: ib` viene usato `IB`.

`transfer_kwargs`
: Mapping. Default: `{}`.
  Argomenti passati all'assemblaggio del transfer. Per mesh Q1:
  `ID` accetta `quadrature_order`;
  `IB` accetta `quadrature_order`;
  `MIX_noC` accetta `id_quadrature_order`, `ib_quadrature_order`;
  `MIX` accetta `id_quadrature_order`, `ib_quadrature_order`,
  `dirichlet_flag`.

`tol`
: Float non negativo. Default: `1.0e-6`.
  Tolleranza del fixed-point coupling.

`tolerance`
: Float non negativo. Alias di `tol`; usato solo se `tol` non e' presente.

`convergence`
: Stringa. Default: `residual`.
  Valori ammessi: `residual`, `lambda`, `force`, `position`,
  `solid_position`, `solid`, `both`, `max`, `either`, `any`, `min`.
  Sceglie quale errore usare per fermare l'iterazione.

`relative_tol`
: Booleano. Default: `true`.
  Se `true`, usa errori relativi. Se `false`, usa errori assoluti.

`residual_reference`
: Stringa. Default: `old_lambda`.
  Valori ammessi: `old_lambda`, `old`, `previous`, `lambda_old`,
  `new_lambda`, `relaxed`, `lambda`, `lambda_new`, `raw_lambda`, `raw`,
  `unrelaxed`, `lambda_raw`, `max`, `largest`.
  Determina il denominatore dell'errore relativo sul residuo.

`norm_floor`
: Float. Default: `1.0`.
  Soglia minima per il denominatore degli errori relativi.

`min_iterations`
: Intero maggiore o uguale a `0`. Default: `0`.
  Numero minimo di iterazioni fixed-point per ogni step.

`max_iterations`
: Intero maggiore o uguale a `1`. Default: `100`.
  Numero massimo di iterazioni fixed-point per ogni step.

`mass_model`
: Stringa. Default: `buoyant`.
  Valori ammessi: `buoyant`, `difference`, `rho_s-rho_f`,
  `rho_s_minus_rho_f`, `solid`, `rho_s`, `unit`, `none`, `matrix`.
  Determina la scala della matrice di massa strutturale effettiva.

`rayleigh_alpha`
: Float non negativo. Default: `0.0`.
  Coefficiente alpha dello smorzamento di Rayleigh.

`rayleigh_beta`
: Float non negativo. Default: `0.0`.
  Coefficiente beta dello smorzamento di Rayleigh.

`raise_on_nonconvergence`
: Booleano. Default: `true`.
  Se `true`, una mancata convergenza ferma la simulazione con errore.
  Se `false`, produce un warning e continua.

### `coupling.aitken`

`enabled`
: Booleano. Default: `true`.
  Attiva/disattiva rilassamento vettoriale di Aitken.

`omega0`
: Float. Default: `0.5`.
  Valore iniziale del fattore di rilassamento.

`omega_min`
: Float positivo. Default: `1.0e-4`.
  Limite inferiore per omega.

`omega_max`
: Float maggiore o uguale a `omega_min`. Default: `1.0`.
  Limite superiore per omega.

## `output`

`folder`
: Stringa o path. Default: `cases/outputs/<case.name>`.
  Cartella di output. Se relativa, viene risolta dalla root del progetto.

`file_base`
: Stringa. Default: `case.name`.
  Base del nome file. Produce `<file_base>.npz` e `<file_base>.csv`.

`save_mode`
: Stringa. Default: `history`.
  Valori ammessi: `history`, `full`, `all`, `final`, `minimal`,
  `final_only`, `last`, `both`, `history_and_final`.
  `history` salva le storie; `final` salva solo l'ultimo frame; `both` salva
  entrambe le forme.

`compressed`
: Booleano. Default: `false`.
  Se `true`, usa `np.savez_compressed`.

`overwrite`
: Booleano. Default: `true`.
  Se `false`, errore quando i file esistono gia'.

`unique`
: Booleano. Default: `false`.
  Se `true`, in caso di file esistente crea un nome numerato.

`store_fluid_history`
: Booleano. Default: `true`.
  Salva la storia dei campi fluidi `u_f`, `v_f`.

`store_fluid_pressure`
: Booleano. Default: `true`.
  Salva la pressione fluida `p_f` quando viene salvata la storia fluida.

`store_solid_history`
: Booleano. Default: `true`.
  Salva la storia degli stati solidi `u_s`, `v_s`, `a_s`.

`store_iteration_history`
: Booleano. Default: `true`.
  Salva tabella delle iterazioni fixed-point.

`history_stride`
: Intero maggiore o uguale a `1`. Default: `1`.
  Alternativa a `time.history_stride`; usata solo se `time.history_stride`
  non e' presente.

`verbose`
: Booleano. Default: `true`.
  Stampa avanzamento della simulazione.

`print_every`
: Intero. Default: `10`.
  Frequenza di stampa degli step quando `verbose: true`.
