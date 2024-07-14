import typer
import ocr_map as om

app = typer.Typer()

@app.callback()
def callback():
  ...

@app.command('fit')
def fit(
  pickle_output: str,
  verbose: bool = typer.Option(False, '-v', '--verbose'),
):
  """Fits a model to samples read from stdin and saves it to a pickle file.
  Expected input format: "{lab}\\t{top_preds_json}" per line.
  """
  if verbose:
    print('Parsing samples from stdin...')
  import sys
  from .parse import parse_lines
  samples = list(parse_lines(sys.stdin))
  
  if verbose:
    print(f'Parsed {len(samples)} samples. Fitting model...')
  model = om.Model.fit(samples)
  if verbose:
    print(f'Saving model to {pickle_output}...')
  import pickle
  with open(pickle_output, 'wb') as f:
    pickle.dump(model, f)