import logging
import pickle
import random
import string
from pathlib import Path

import click
import h5py
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from PIL import Image

from .data import (
    Creds,
    Database,
    GeodatabaseImporter,
    ImageCropper,
    ImportManager,
    InspectionNotesImporter,
    MatrixCreator,
    count_datasets_in_hdf5,
    fetch_all_blocklots,
    fetch_blocklots_imaged,
    fetch_image_from_hdf5,
    fetch_image_predictions,
)
from .modeling import (
    get_modeling_status,
    train_image_model,
    train_many_models,
    fetch_labels,
)
from .modeling.image_model import ImageModel
from .modeling.models import write_completed_preds_to_db
from .reporting import Reporter, evaluate

load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

this_dir = Path(__file__).resolve().parent
SEED = 0


@click.group()
@click.option("--user", envvar="PGUSER")
@click.option("--password", envvar="PGPASSWORD")
@click.option("--host", envvar="PGHOST")
@click.option("--port", envvar="PGPORT")
@click.option("--database", envvar="PGDATABASE")
@click.pass_context
def roofs(ctx, user, password, host, port, database):
    """Command line interface tools for setting up and detecting roof damage"""
    creds = Creds(user, password, host, port, database)
    ctx.obj = {"db": Database(creds)}


@roofs.group()
def db():
    """Tools for loading and maintaining the database"""
    pass


@roofs.group()
@click.option(
    "--models-path",
    "-m",
    type=click.Path(file_okay=False, dir_okay=True, exists=True, path_type=Path),
    help="path where models are stored",
    default=this_dir.parent / "baltimoreroofs/modeling",
)
@click.pass_context
def train(ctx, models_path):
    """Tools for training roof damage classification models"""
    ctx.obj["models_path"] = models_path


@train.command(name="status")
@click.argument(
    "hdf5",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
)
@click.pass_obj
def modeling_status(obj, hdf5):
    """Status of the training pipeline

    HDF5 is the path to the hdf5 file containing blocklot images.
    """
    db, models_path = obj["db"], obj["models_path"]
    click.echo(get_modeling_status(db, models_path, hdf5))


@train.command(name="image-model")
@click.argument(
    "hdf5",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
)
@click.option("--seed", default=SEED, type=int)
@click.pass_obj
def cli_train_image_model(obj, hdf5, seed):
    """Train an image classification model from aerial photos

    HDF5 is the path to the hdf5 file containing blocklot images.
    """
    train_image_model(obj["db"], obj["models_path"], hdf5, seed=seed)


@train.command()
@click.argument(
    "hdf5",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
)
@click.argument(
    "output",
    type=click.Path(file_okay=True, dir_okay=False, writable=True, path_type=Path),
)
@click.option(
    "--model-path",
    "-m",
    help="directory to save models into",
    default=Path(__file__).resolve().parent.parent / "models",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "--max-date",
    "-d",
    help="Most recent date to consider",
    default="2023-01-03",
    type=click.DateTime(),
)
@click.pass_obj
def model(obj, hdf5, output, model_path, max_date):
    """Train a new model to classify roof damage severity

    HDF5 is the path to the hdf5 file containing blocklot images.
    OUTPUT is the filename of the CSV into which to write scores

    In addition to the CSV, a PNG of the same name will be written
    that graphs the scores.
    """
    df, fig = train_many_models(obj["db"], model_path, hdf5, max_date)
    graph_filename = output.with_suffix(".png")
    fig.savefig(graph_filename, dpi=300)
    df.to_csv(output)
    click.echo(f"Wrote scores to {output} and graph to {graph_filename}")


@roofs.group()
@click.option(
    "--models-path",
    "-m",
    type=click.Path(file_okay=False, dir_okay=True, exists=True, path_type=Path),
    help="path where models are stored",
    default=this_dir.parent / "models",
)
@click.pass_context
def report(ctx, models_path):
    """Write out reports using trained models"""
    ctx.obj["models_path"] = models_path


@report.command()
@click.argument(
    "model_path",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
)
@click.argument(
    "output",
    type=click.Path(file_okay=True, dir_okay=False, exists=False, path_type=Path),
)
@click.argument(
    "hdf5",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
)
@click.option("--blocklots", "-b", type=str, multiple=True, default=[])
@click.option(
    "--max-date",
    "-d",
    help="Most recent date to consider",
    default="2023-01-03",
    type=click.DateTime(),
)
@click.pass_obj
def predictions(obj, model_path, output, hdf5, blocklots, max_date):
    """Generate roof damage scores from a given model

    MODEL_PATH is the path to a trained model

    OUTPUT is the path at which the output CSV of predictions should be written

    HDF5 is the path to the hdf5 file containing blocklot images
    """
    if len(blocklots) == 0:
        with h5py.File(hdf5) as f:
            blocklots = fetch_blocklots_imaged(f)
    reporter = Reporter(obj["db"])
    preds = reporter.predictions(model_path, hdf5, blocklots, max_date)
    labels = fetch_labels(obj["db"])
    df = pd.DataFrame(preds, index=blocklots, columns=["damage_score"])
    df.index.name = "blocklot"
    df["damage_score"] = preds
    df["label"] = labels
    df["codemap"] = pd.Series(reporter.codemap_urls(df.index))
    df["codemap_ext"] = pd.Series(reporter.codemap_ext_urls(df.index))
    df["pictometry"] = pd.Series(reporter.pictometry_urls(df.index))
    df.sort_values("damage_score", ascending=False, inplace=True)
    df.to_csv(output)
    print(f"Wrote predictions to {output}")


@report.command()
@click.argument(
    "model_path",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
)
@click.argument(
    "output",
    type=click.Path(file_okay=True, dir_okay=False, exists=False, path_type=Path),
)
@click.argument(
    "hdf5",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
)
@click.option(
    "--max-date",
    "-d",
    help="Most recent date to consider",
    default="2023-01-03",
    type=click.DateTime(),
)
@click.option(
    "--top-k",
    "-k",
    help="Run evaluations at this top-k value",
    multiple=True,
    default=[1000, 2500, 3000, 5000],
    type=int,
)
@click.option(
    "--eval-test-only/--eval-on-all",
    default=False,
    help="Evaluate the model performance only on the test set",
)
@click.pass_obj
def evals(obj, model_path, output, hdf5, max_date, top_k, eval_test_only):
    """Evaluate the performance of a given model

    MODEL_PATH is the path to a trained model
    HDF5 is the path to the hdf5 file containing blocklot images

    The --eval-test-only option is useful for gauging the performance of the model
    on unseen data. It will run an evaluation only on labeled data that the model
    did not see during training (assuming the same seed). Without the option,
    performance will be evaluated on all data, which will include data that was in
    the training set. This means that if the model overfit to the training data,
    the performance will look better than you'll be able to expect on new data.
    """
    threshold_counts, top_k_counts = evaluate(
        obj["db"], model_path, hdf5, max_date, top_k, eval_test_only
    )
    # graph_filename = output.with_suffix(".png")
    # fig.savefig(graph_filename, dpi=300)
    # print(threshold_counts)
    df = pd.concat(
        [
            pd.DataFrame.from_dict(top_k_counts, orient="index"),
            pd.DataFrame.from_dict(threshold_counts, orient="index"),
        ]
    )
    df.to_csv(output)
    click.echo(f"Wrote scores to {output}")  # and graph to {graph_filename}")


@report.command()
@click.argument(
    "preds",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path
    ),
)
@click.argument(
    "image-root",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, readable=True, path_type=Path
    ),
)
@click.argument("output", type=click.File(mode="w", encoding="utf-8"))
@click.option(
    "--top-n",
    type=int,
    help="How many blocklots to add to report from the prediction CSV",
    default=100,
)
@click.pass_obj
def html(obj, preds, image_root, output, top_n):
    """Generate an HTML report of predictions

    PREDS is the predictions CSV that results from `roofs report predictions`
    IMAGE_ROOT is the root directory of the aerial images"""
    report = Reporter(obj["db"]).html_report(preds, image_root, top_n)
    output.write(report)
    click.echo(f"Wrote HTML report to {output.name}")


@roofs.group()
@click.pass_context
def images(ctx):
    """Tools for interacting with aerial images"""
    pass


@images.command()
@click.argument(
    "image-root",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.argument(
    "output",
    type=click.Path(file_okay=True, dir_okay=False, writable=True, path_type=Path),
)
@click.option("--blocklots", "-b", type=str, multiple=True, default=[])
@click.option("--overwrite/--no-overwrite", default=False)
@click.pass_obj
def crop(obj, image_root, output, blocklots, overwrite):
    """Crop aerial image tiles into individual blocklot images

    IMAGE_ROOT is the path of the directory containing all the .tif images
    and their accompanying .tif.aux.xml files.

    OUTPUT is the path of the hdf5 file that will be created containing
    the blocklot images.
    """
    db = obj["db"]
    cropper = ImageCropper(db, image_root)
    if len(blocklots) == 0:
        blocklots = fetch_all_blocklots(db, db.CLEAN_SCHEMA)
    cropper.write_h5py(output, blocklots, overwrite)


@images.command(name="predict")
@click.argument(
    "hdf5",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
)
@click.option(
    "--image-model",
    "-m",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
    help="path to image model",
    default=this_dir.parent / "models" / "image_model.pth",
)
@click.pass_obj
def images_predict(obj, hdf5, image_model):
    """Make predictions using just the image model"""
    db = obj["db"]
    model = ImageModel.load(image_model, hdf5)
    click.echo(f"Running on {model.device}")
    with h5py.File(hdf5) as f:
        blocklots = fetch_blocklots_imaged(f)
    preds = model.forward(blocklots)
    write_completed_preds_to_db(db, "image_model", preds)
    click.echo(f"Wrote {len(preds):,} predictions to database.")


@images.command(name="status")
@click.argument(
    "hdf5",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
)
@click.pass_obj
def images_status(obj, hdf5):
    """Show the status of the blocklot image setup process

    hdf5 is the path to the hdf5 file containing all the blocklot image data.
    """
    db = obj["db"]
    blocklots = fetch_all_blocklots(db, db.CLEAN_SCHEMA)
    click.echo("Images Status\n" + "=" * 50)
    click.echo(f"There are {len(blocklots):,} blocklots in the database.")
    click.echo(f"    Here are a few: {random.sample(blocklots, k=3)}")
    with h5py.File(hdf5) as f:
        n_datasets = count_datasets_in_hdf5(f)
        click.echo(f"\nThere are {n_datasets:,} blocklot images in the image database.")
        sample = random.sample(fetch_blocklots_imaged(f), k=3)
        click.echo(f"    Here are a few: {sample}")
    # TODO need to check if any predictions have been done yet otherwise this next part will fail
    # image_preds = fetch_image_predictions(db, blocklots)
    # click.echo(
    #     f"There are {len(image_preds):,} predictions from the image model "
    #     "in the database."
    # )


@images.command
@click.argument(
    "hdf5",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
)
@click.argument(
    "output",
    type=click.Path(file_okay=False, dir_okay=True, writable=True, path_type=Path),
)
@click.option("--blocklots", "-b", type=str, multiple=True, required=True)
def dump(hdf5, output, blocklots):
    """Dump JPEG images of blocklots to disk for further inspection

    HDF5 is the path to the hdf5 file containing blocklot images.
    OUTPUT is the directory into which images will be written.
    """

    with h5py.File(hdf5) as f:
        for blocklot in blocklots:
            array = fetch_image_from_hdf5(blocklot, f)
            pixels = np.nan_to_num(array[:], nan=255).astype("uint8")
            im = Image.fromarray(pixels)
            im.save(output / f"{blocklot}.jpeg")


@db.command()
@click.option(
    "-i",
    "--inspection-notes",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.pass_obj
def import_sheet(
    obj,
    inspection_notes: Path | None = None,
):
    """Import spreadsheets of data

    Only handles inspection notes currently."""
    db = obj["db"]
    importers = []
    if inspection_notes:
        importers.append(InspectionNotesImporter.from_file(db, inspection_notes))
    manager = ImportManager(db, importers)
    manager.setup()
    click.echo(manager.status())


@db.command()
@click.argument(
    "geodatabase",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.option("--building-outlines", help="layer name that shows building outlines")
@click.option("--building-permits", help="layer name that has construction permits")
@click.option("--code-violations", help="layer name that has building code violations")
@click.option("--data-311", help="layer name that contains 311 data")
@click.option("--demolitions", help="layer name that contains demolitions data")
@click.option("--ground-truth", help="layer name that includes roof quality labels")
@click.option("--real-estate", help="layer name that includes real estate transactions")
@click.option("--redlining", help="layer name that shows historical redlines")
@click.option(
    "--tax-parcel-address", help="layer name that contains tax parcel addresses"
)
@click.option(
    "--vacant-building-notices",
    help="layer name that contains the vacant building notices",
)
@click.pass_obj
def import_gdb(obj, geodatabase: Path, **layer_map):
    """Import a Geodatabase file"""
    db = obj["db"]
    if all(layer_map[layer] is None for layer in GeodatabaseImporter.REQUIRED_TABLES):
        raise click.UsageError("At least one layer name option must be specified.")
    layer_map = {
        table: layer for table, layer in layer_map.items() if layer is not None
    }
    importer = GeodatabaseImporter.from_file(db, geodatabase, layer_map)
    manager = ImportManager(db, [importer])
    manager.setup()
    click.echo(manager.status())


@db.command(name="status")
@click.pass_obj
def db_status(obj):
    """Show the status of the database"""
    db = obj["db"]
    click.echo(ImportManager(db).status())
    try:
        blocklots = fetch_all_blocklots(db, db.CLEAN_SCHEMA)
        click.echo(f"There are {len(blocklots):,} row home blocklots in the database.")
    except Exception:
        pass


@db.command()
@click.confirmation_option(prompt="Are you sure you want to reset the database?")
@click.pass_obj
def reset(obj):
    """Remove all data from the database"""
    db = obj["db"]
    ImportManager(db).reset()
    click.echo("The database has been reset.")


@db.command()
@click.option(
    "--min-demo-date",
    "-d",
    help="Exclude blocklots that have had a demo after this date",
    default="2014-01-01",  # Picked just by looking at aerial images of what is built and when demos took place
    type=click.DateTime(),
)
@click.pass_obj
def filter(obj, min_demo_date):
    """Filter the ground truth to just the row homes we're interested in

    The label date option makes sure that blocklots that have
    seen a demolition after damage was labeled doesn't make it
    into the dataset."""
    db = obj["db"]
    click.echo("Filtering blocklots...")
    GeodatabaseImporter(db).filter_to_cohort(min_demo_date)
    click.echo("Filtered blocklots.")


@roofs.group()
def misc():
    """Miscellaneous helpful widgets"""
    pass


@misc.command()
@click.argument(
    "output",
    type=click.Path(file_okay=True, dir_okay=False, exists=False, path_type=Path),
)
@click.argument(
    "sheets",
    nargs=-1,
    type=click.Path(file_okay=True, dir_okay=False, readable=True, path_type=Path),
)
@click.option(
    "--suffixes", multiple=True, help="Suffixes for the fields from each of the sheets"
)
def merge_sheets(output, sheets, suffixes):
    """Merge a number of CSVs or Excel files on blocklot

    All sheets must have a column titled "blocklot"."""
    dfs = []
    if len(suffixes) == 0:
        suffixes = [f"{letter}_" for letter in string.ascii_lowercase]

    for sheet in sheets:
        if ".xls" in str(sheet):
            df = pd.read_excel(sheet)
        else:
            df = pd.read_csv(sheet)
        df["blocklot"] = df["blocklot"].astype(str)
        dfs.append(df)

    merged = dfs[0]
    for i, df in enumerate(dfs[1:]):
        merged = pd.merge(
            merged,
            df,
            on="blocklot",
            how="outer",
            suffixes=(suffixes[0], suffixes[i + 1]),
        )
    if "damaged" in merged.columns:
        merged = merged.sort_values("damaged", ascending=False)
    merged.to_csv(output, index=False)
    print(f"Merged {len(sheets)} sheets and wrote to {output}")


if __name__ == "__main__":
    roofs()
