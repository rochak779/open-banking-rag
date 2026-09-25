import textwrap

import pytest

from obrag.ingest.legislation import parse_legislation_file

# Mirrors real legislation.gov.uk CLML: provisions carry id and DocumentURI,
# amended provisions have a CommentaryRef inside Pnumber, amended words sit in
# <Substitution> inside <Text>, schedules nest paragraphs under Part, and
# BlockAmendment quotes provisions of *other* instruments.
CLML = textwrap.dedent(
    """
    <Legislation xmlns="http://www.legislation.gov.uk/namespaces/legislation">
      <Secondary>
        <Body>
          <Part>
            <Number>PART 7</Number>
            <Title>Rights and obligations</Title>
            <P1group>
              <Title>Obligation to provide information</Title>
              <P1 id="regulation-68" DocumentURI="http://www.legislation.gov.uk/uksi/2017/752/regulation/68">
                <Pnumber>68</Pnumber>
                <P1para>
                  <P2 id="regulation-68-1">
                    <Pnumber>1</Pnumber>
                    <P2para><Text>A payment service provider must provide the information.</Text></P2para>
                  </P2>
                  <P2 id="regulation-68-2">
                    <Pnumber>2</Pnumber>
                    <P2para>
                      <Text>The conditions are that&#8212;</Text>
                      <P3 id="regulation-68-2-a">
                        <Pnumber>a</Pnumber>
                        <P3para><Text>it complies with the <Substitution>technical standards</Substitution> made under regulation 106A.</Text></P3para>
                      </P3>
                    </P2para>
                  </P2>
                </P1para>
              </P1>
            </P1group>
            <P1group>
              <Title>Conditions for authorisation</Title>
              <P1 id="regulation-6" DocumentURI="http://www.legislation.gov.uk/uksi/2017/752/regulation/6">
                <Pnumber><CommentaryRef Ref="key-1"/>6</Pnumber>
                <P1para>
                  <Text>The FCA may refuse an application.</Text>
                  <P2 id="regulation-6-2">
                    <Pnumber>2</Pnumber>
                    <Substitution>A wholly substituted paragraph outside any Text element.</Substitution>
                  </P2>
                </P1para>
              </P1>
            </P1group>
          </Part>
        </Body>
        <Schedules>
          <Schedule id="schedule-1">
            <Number>SCHEDULE 1</Number>
            <TitleBlock><Title>Payment Services</Title></TitleBlock>
            <ScheduleBody>
              <Part id="schedule-1-part-2">
                <Number>PART 2</Number>
                <Title>Activities which do not constitute payment services</Title>
                <P1 id="schedule-1-paragraph-2" DocumentURI="http://www.legislation.gov.uk/uksi/2017/752/schedule/1/paragraph/2">
                  <Pnumber>2</Pnumber>
                  <P1para>
                    <Text>The following do not constitute payment services.</Text>
                    <Tabular><table><tbody><tr><td>Table cell text</td></tr></tbody></table></Tabular>
                    <BlockAmendment>
                      <P1><Pnumber>5A</Pnumber><P1para><Text>Quoted text of another Act.</Text></P1para></P1>
                    </BlockAmendment>
                  </P1para>
                </P1>
              </Part>
            </ScheduleBody>
          </Schedule>
        </Schedules>
      </Secondary>
    </Legislation>
    """
).strip()


@pytest.fixture
def chunks(tmp_path):
    path = tmp_path / "psr_2017.xml"
    path.write_text(CLML)
    return {c.metadata["provision_id"]: c for c in parse_legislation_file(path, title="PSR 2017")}


def test_one_chunk_per_provision_excluding_quoted_amendments(chunks):
    assert sorted(chunks) == ["regulation-6", "regulation-68", "schedule-1-paragraph-2"]


def test_chunks_are_in_the_regulation_collection(chunks):
    assert {c.collection for c in chunks.values()} == {"regulation"}


def test_citation_names_the_instrument_and_provision(chunks):
    assert chunks["regulation-68"].citation == "PSR 2017, regulation 68"
    assert chunks["schedule-1-paragraph-2"].citation == "PSR 2017, Schedule 1, paragraph 2"


def test_amended_provision_with_commentary_ref_in_number_is_kept(chunks):
    assert chunks["regulation-6"].citation == "PSR 2017, regulation 6"
    assert "The FCA may refuse an application." in chunks["regulation-6"].text


def test_text_includes_headings_and_every_subparagraph_in_order(chunks):
    text = chunks["regulation-68"].text
    assert "Rights and obligations" in text
    assert "Obligation to provide information" in text
    assert "(1) A payment service provider must provide the information." in text
    assert "(2) The conditions are that" in text
    assert "(a) it complies with the technical standards made under regulation 106A." in text
    assert text.index("(1)") < text.index("(2)") < text.index("(a)")


def test_schedule_paragraph_carries_schedule_and_part_titles(chunks):
    chunk = chunks["schedule-1-paragraph-2"]
    assert "Payment Services" in chunk.text
    assert "Activities which do not constitute payment services" in chunk.text
    assert "Quoted text of another Act." in chunk.text  # quoted amendment stays inside its parent


def test_source_url_is_the_provisions_own_https_document_uri(chunks):
    assert chunks["regulation-68"].source_url == (
        "https://www.legislation.gov.uk/uksi/2017/752/regulation/68"
    )
    assert chunks["schedule-1-paragraph-2"].source_url == (
        "https://www.legislation.gov.uk/uksi/2017/752/schedule/1/paragraph/2"
    )


def test_ids_are_stable_and_unique(chunks):
    assert chunks["regulation-68"].id == "reg:psr_2017:regulation-68"
    assert len({c.id for c in chunks.values()}) == len(chunks)


def test_empty_provisions_are_skipped(tmp_path):
    empty = tmp_path / "empty.xml"
    empty.write_text(
        '<Legislation xmlns="http://www.legislation.gov.uk/namespaces/legislation">'
        '<Secondary><Body><P1group><P1 id="regulation-1" DocumentURI="http://x/regulation/1">'
        "<Pnumber>1</Pnumber></P1></P1group></Body></Secondary></Legislation>"
    )
    assert parse_legislation_file(empty, title="X") == []


def test_text_outside_text_elements_is_not_dropped(chunks):
    assert "(2) A wholly substituted paragraph outside any Text element." in chunks["regulation-6"].text
    assert "Table cell text" in chunks["schedule-1-paragraph-2"].text
