import os
from pathlib import Path
from typing import Dict

import pytest
from langchain_core.documents import Document
from pytest_mock import MockerFixture

from langchain_community.document_loaders import CSVLoader, PyPDFLoader

EXAMPLE_DOCS_DIRECTORY = str(Path(__file__).parent.parent.parent / "examples/")


class MockResponse:
    def __init__(self, json_data: Dict, status_code: int):
        self.json_data = json_data
        self.status_code = status_code

    def json(self) -> Dict:
        return self.json_data


def test_pebblo_import() -> None:
    """Test that the Pebblo safe loader can be imported."""
    from langchain_community.document_loaders import PebbloSafeLoader  # noqa: F401


def test_empty_filebased_loader(mocker: MockerFixture) -> None:
    """Test basic file based csv loader."""
    # Setup
    from langchain_community.document_loaders import PebbloSafeLoader

    mocker.patch.multiple(
        "requests",
        get=MockResponse(json_data={"data": ""}, status_code=200),
        post=MockResponse(json_data={"data": ""}, status_code=200),
    )

    file_path = os.path.join(EXAMPLE_DOCS_DIRECTORY, "test_empty.csv")
    expected_docs: list = []

    # Exercise
    loader = PebbloSafeLoader(
        CSVLoader(file_path=file_path),
        "dummy_app_name",
        "dummy_owner",
        "dummy_description",
    )
    result = loader.load()

    # Assert
    assert result == expected_docs


def test_csv_loader_load_valid_data(mocker: MockerFixture) -> None:
    # Setup
    from langchain_community.document_loaders import PebbloSafeLoader

    mocker.patch.multiple(
        "requests",
        get=MockResponse(json_data={"data": ""}, status_code=200),
        post=MockResponse(json_data={"data": ""}, status_code=200),
    )
    file_path = os.path.join(EXAMPLE_DOCS_DIRECTORY, "test_nominal.csv")
    expected_docs = [
        Document(
            page_content="column1: value1\ncolumn2: value2\ncolumn3: value3",
            metadata={"source": file_path, "row": 0},
        ),
        Document(
            page_content="column1: value4\ncolumn2: value5\ncolumn3: value6",
            metadata={"source": file_path, "row": 1},
        ),
    ]

    # Exercise
    loader = PebbloSafeLoader(
        CSVLoader(file_path=file_path),
        "dummy_app_name",
        "dummy_owner",
        "dummy_description",
    )
    result = loader.load()

    # Assert
    assert result == expected_docs


@pytest.mark.requires("pypdf")
def test_pdf_lazy_load(mocker: MockerFixture) -> None:
    # Setup
    from langchain_community.document_loaders import PebbloSafeLoader

    mocker.patch.multiple(
        "requests",
        get=MockResponse(json_data={"data": [Document(
        page_content="Shape helpers\nPDFKit also includes some helpers that make defining common shapes much easier. Here is \na list of the helpers.\nrect(x, y, width, height)\nroundedRect(x, y, width, height, cornerRadius)\nellipse(centerX, centerY, radiusX, radiusY = radiusX)\ncircle(centerX, centerY, radius)\npolygon(points...)\nThe last one, polygon, allows you to pass in a list of points (arrays of x,y pairs), and it will \ncreate the shape by moving to the first point, and then drawing lines to each consecutive \npoint. Here is how you'd draw a triangle with the polygon helper.\ndoc.polygon([100, 0], [50, 100], [150, 100]);\ndoc.stroke();\nThe output of this example looks like this:",
        metadata={
            "source": "https://drive.google.com/file/d/1LqT7I7l6-FFDSG0GzShomM2UOIGuKAR-/view",
            "title": "guide.pdf",
            "page": 14,
        },
    ),

    Document(
        page_content="\n\n\nName\nEmail\nSSN\nAddress\nCC Expiry\nCredit Card Number\nCC Security Code\nIPv4\nIPv6\nPhone\n\n\nxGelEeIfPW\njDhHvGhCQM@IlJqV.com\n265923644\nBLQvsSCvuqiMcZyMScwJ\n11/2025\n6267494999707042\n423\n7.178.156.177\n9b37:ec97:c3d0:d7ab:cda8:539f:9cc1:67fa\n2690137480\n\n\n",
        metadata={
            "text_as_html": '<table border="1" class="dataframe">\n  <tbody>\n    <tr>\n      <td>Name</td>\n      <td>Email</td>\n      <td>SSN</td>\n      <td>Address</td>\n      <td>CC Expiry</td>\n      <td>Credit Card Number</td>\n      <td>CC Security Code</td>\n      <td>IPv4</td>\n      <td>IPv6</td>\n      <td>Phone</td>\n    </tr>\n    <tr>\n      <td>xGelEeIfPW</td>\n      <td>jDhHvGhCQM@IlJqV.com</td>\n      <td>265923644</td>\n      <td>BLQvsSCvuqiMcZyMScwJ</td>\n      <td>11/2025</td>\n      <td>6267494999707042</td>\n      <td>423</td>\n      <td>7.178.156.177</td>\n      <td>9b37:ec97:c3d0:d7ab:cda8:539f:9cc1:67fa</td>\n      <td>2690137480</td>\n    </tr>\n  </tbody>\n</table>',
            "languages": ["eng", "cat"],
            "filetype": "text/csv",
            "category": "Table",
            "source": "https://drive.google.com/file/d/1-GzHYOensjcG0yey8nRMlY-QpZVm4zdp/view",
            "title": "sens_data.csv",
        },
    ),  ]}, status_code=200),
        post=MockResponse(json_data={"data": [Document(
        page_content="Shape helpers\nPDFKit also includes some helpers that make defining common shapes much easier. Here is \na list of the helpers.\nrect(x, y, width, height)\nroundedRect(x, y, width, height, cornerRadius)\nellipse(centerX, centerY, radiusX, radiusY = radiusX)\ncircle(centerX, centerY, radius)\npolygon(points...)\nThe last one, polygon, allows you to pass in a list of points (arrays of x,y pairs), and it will \ncreate the shape by moving to the first point, and then drawing lines to each consecutive \npoint. Here is how you'd draw a triangle with the polygon helper.\ndoc.polygon([100, 0], [50, 100], [150, 100]);\ndoc.stroke();\nThe output of this example looks like this:",
        metadata={
            "source": "https://drive.google.com/file/d/1LqT7I7l6-FFDSG0GzShomM2UOIGuKAR-/view",
            "title": "guide.pdf",
            "page": 14,
        },
    ),

    Document(
        page_content="\n\n\nName\nEmail\nSSN\nAddress\nCC Expiry\nCredit Card Number\nCC Security Code\nIPv4\nIPv6\nPhone\n\n\nxGelEeIfPW\njDhHvGhCQM@IlJqV.com\n265923644\nBLQvsSCvuqiMcZyMScwJ\n11/2025\n6267494999707042\n423\n7.178.156.177\n9b37:ec97:c3d0:d7ab:cda8:539f:9cc1:67fa\n2690137480\n\n\n",
        metadata={
            "text_as_html": '<table border="1" class="dataframe">\n  <tbody>\n    <tr>\n      <td>Name</td>\n      <td>Email</td>\n      <td>SSN</td>\n      <td>Address</td>\n      <td>CC Expiry</td>\n      <td>Credit Card Number</td>\n      <td>CC Security Code</td>\n      <td>IPv4</td>\n      <td>IPv6</td>\n      <td>Phone</td>\n    </tr>\n    <tr>\n      <td>xGelEeIfPW</td>\n      <td>jDhHvGhCQM@IlJqV.com</td>\n      <td>265923644</td>\n      <td>BLQvsSCvuqiMcZyMScwJ</td>\n      <td>11/2025</td>\n      <td>6267494999707042</td>\n      <td>423</td>\n      <td>7.178.156.177</td>\n      <td>9b37:ec97:c3d0:d7ab:cda8:539f:9cc1:67fa</td>\n      <td>2690137480</td>\n    </tr>\n  </tbody>\n</table>',
            "languages": ["eng", "cat"],
            "filetype": "text/csv",
            "category": "Table",
            "source": "https://drive.google.com/file/d/1-GzHYOensjcG0yey8nRMlY-QpZVm4zdp/view",
            "title": "sens_data.csv",
        },
    ),]}, status_code=200),
    )
    file_path = os.path.join(
        EXAMPLE_DOCS_DIRECTORY, "multi-page-forms-sample-2-page.pdf"
    )

    # Exercise
    loader = PebbloSafeLoader(
        PyPDFLoader(file_path=file_path),
        "dummy_app_name",
        "dummy_owner",
        "dummy_description",
    )

    result = list(loader.lazy_load())

    # Assert
    assert len(result) == 2


def test_pebblo_safe_loader_api_key() -> None:
    # Setup
    from langchain_community.document_loaders import PebbloSafeLoader

    file_path = os.path.join(EXAMPLE_DOCS_DIRECTORY, "test_empty.csv")
    api_key = "dummy_api_key"

    # Exercise
    loader = PebbloSafeLoader(
        CSVLoader(file_path=file_path),
        "dummy_app_name",
        "dummy_owner",
        "dummy_description",
        api_key=api_key,
    )

    # Assert
    assert loader.api_key == api_key
