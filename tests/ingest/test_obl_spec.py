import textwrap

import pytest

from obrag.ingest.obl_spec import parse_openapi_file, parse_spec_dir

# Mirrors the real OBL layout: parameters and responses are $refs into
# components, and each body is repeated under several media types.
SAMPLE = textwrap.dedent(
    """
    openapi: 3.0.1
    info:
      title: Account and Transaction API Specification
      version: 4.0.1
    paths:
      /accounts:
        get:
          operationId: GetAccounts
          summary: Get Accounts
          description: Retrieve the list of accounts the consent covers.
          parameters:
            - $ref: '#/components/parameters/x-fapi-interaction-id'
          responses:
            '200':
              $ref: '#/components/responses/200AccountsRead'
            '403':
              $ref: '#/components/responses/403Error'
    components:
      parameters:
        x-fapi-interaction-id:
          name: x-fapi-interaction-id
          in: header
          required: false
          description: An RFC4122 UID used as a correlation ID.
      responses:
        200AccountsRead:
          description: Accounts successfully read
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/OBReadAccount6'
            application/jose+jwe:
              schema:
                $ref: '#/components/schemas/OBReadAccount6'
        403Error:
          description: Forbidden
      schemas:
        OBReadAccount6:
          description: A list of accounts and their identification details.
          type: object
    """
).strip()

TAG = "v4.0.1-Update-1"


@pytest.fixture
def spec_file(tmp_path):
    openapi_dir = tmp_path / f"obl-specs-{TAG}" / "dist" / "openapi"
    openapi_dir.mkdir(parents=True)
    path = openapi_dir / "account-info.yaml"
    path.write_text(SAMPLE)
    return path


def test_one_chunk_per_operation(spec_file):
    chunks = parse_openapi_file(spec_file)
    assert len(chunks) == 1


def test_chunk_is_in_the_spec_collection(spec_file):
    assert parse_openapi_file(spec_file)[0].collection == "spec"


def test_citation_is_the_method_and_path(spec_file):
    chunk = parse_openapi_file(spec_file)[0]
    assert chunk.citation == "Account and Transaction API Specification v4.0.1 — GET /accounts"


def test_text_contains_summary_description_params_and_responses(spec_file):
    text = parse_openapi_file(spec_file)[0].text
    assert "Get Accounts" in text
    assert "Retrieve the list of accounts" in text
    assert "x-fapi-interaction-id (in header, optional)" in text
    assert "200: Accounts successfully read" in text
    assert "403: Forbidden" in text
    assert "OBReadAccount6" in text
    assert "A list of accounts and their identification details." in text


def test_schema_repeated_across_media_types_is_listed_once(spec_file):
    text = parse_openapi_file(spec_file)[0].text
    assert text.count("OBReadAccount6") == 1


def test_id_is_stable_and_unique(spec_file):
    chunk = parse_openapi_file(spec_file)[0]
    assert chunk.id == "spec:account-info:get:/accounts"
    assert parse_openapi_file(spec_file)[0].id == chunk.id


def test_source_url_links_to_the_pinned_tag_not_the_version(spec_file):
    chunk = parse_openapi_file(spec_file)[0]
    assert chunk.source_url == (
        "https://github.com/OpenBankingUK/read-write-api-specs"
        f"/blob/{TAG}/dist/openapi/account-info.yaml"
    )


def test_metadata_records_api_and_operation(spec_file):
    meta = parse_openapi_file(spec_file)[0].metadata
    assert meta["method"] == "GET"
    assert meta["path"] == "/accounts"
    assert meta["operation_id"] == "GetAccounts"


def test_parse_spec_dir_reads_every_yaml(spec_file):
    (spec_file.parent / "other.yaml").write_text(SAMPLE)
    chunks = parse_spec_dir(spec_file.parent)
    assert len(chunks) == 2
    assert len({c.id for c in chunks}) == 2
