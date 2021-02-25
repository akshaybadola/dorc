# Adapted from sphinxcontrib.napoleon
# See https://github.com/sphinx-contrib/napoleon

import inspect
import re
from typing import Any, Dict, List, Tuple, Type, Union, Callable
from functools import partial

from six import string_types, u
from six.moves import range

from sphinx.locale import _

from pockets import modify_iter, UnicodeMixin


_directive_regex = re.compile(r'\.\. \S+::')
_google_section_regex = re.compile(r'^(\s|\w)+:\s*$')
_google_typed_arg_regex = re.compile(r'\s*(.+?)\s*\(\s*(.*[^\s]+)\s*\)')
_numpy_section_regex = re.compile(r'^[=\-`:\'"~^_*+#<>]{2,}\s*$')
_single_colon_regex = re.compile(r'(?<!:):(?!:)')
_xref_regex = re.compile(r'(:(?:[a-zA-Z0-9]+[\-_+:.])*[a-zA-Z0-9]+:`.+?`)')
_bullet_list_regex = re.compile(r'^(\*|\+|\-)(\s+\S|\s*$)')
_enumerated_list_regex = re.compile(
    r'^(?P<paren>\()?'
    r'(\d+|#|[ivxlcdm]+|[IVXLCDM]+|[a-zA-Z])'
    r'(?(paren)\)|\.)(\s+\S|\s*$)')


def _ref_repl(x):
    return re.sub(r'(.+)(:[a-zA-Z0-9]+[\-_+:.])`(.+?)`', r'\3', x)


class GoogleDocstring(UnicodeMixin):
    """Convert Google style docstrings to reStructuredText.
    Parameters
    ----------
    docstring : :obj:`str` or :obj:`list` of :obj:`str`
        The docstring to parse, given either as a string or split into
        individual lines.
    Other Parameters
    ----------------
    what : :obj:`str`, optional
        A string specifying the type of the object to which the docstring
        belongs. Valid values: "module", "class", "exception", "function",
        "method", "attribute".
    name : :obj:`str`, optional
        The fully qualified name of the object.
    obj : module, class, exception, function, method, or attribute
        The object to which the docstring belongs.
    options : :class:`sphinx.ext.autodoc.Options`, optional
        The options given to the directive: an object with attributes
        inherited_members, undoc_members, show_inheritance and noindex that
        are True if the flag option of same name was given to the auto
        directive.
    Example
    -------
    >>> from sphinxcontrib.napoleon import Config
    >>> config = Config(napoleon_use_param=True, napoleon_use_rtype=True)
    >>> docstring = '''One line summary.
    ...
    ... Extended description.
    ...
    ... Args:
    ...   arg1(int): Description of `arg1`
    ...   arg2(str): Description of `arg2`
    ... Returns:
    ...   str: Description of return value.
    ... '''

    """

    _name_rgx = re.compile(r"^\s*(:(?P<role>\w+):`(?P<name>[a-zA-Z0-9_.-]+)`|"
                           r" (?P<name2>[a-zA-Z0-9_.-]+))\s*", re.X)

    def __init__(self, docstring: Union[str, List[str]], what: str = '', name: str = '',
                 obj: Any = None, options: Any = None):
        if not what:
            if inspect.isclass(obj):
                what = 'class'
            elif inspect.ismodule(obj):
                what = 'module'
            elif isinstance(obj, Callable):  # type: ignore
                what = 'function'
            else:
                what = 'object'
        self._what = what
        self._name = name
        self._obj = obj
        self._opt = options
        if isinstance(docstring, string_types):
            docstring = docstring.splitlines()
        self._lines = docstring
        self._line_iter = modify_iter(docstring, modifier=lambda s: s.rstrip())
        self._parsed_lines: List[str] = []
        self._is_in_section = False
        self._section_indent = 0
        if not hasattr(self, '_sections'):
            self._sections: Dict[str, Callable] = {
                'args': self._parse_parameters_section,
                'arguments': self._parse_parameters_section,
                'attributes': self._parse_attributes_section,
                'example': self._parse_examples_section,
                'examples': self._parse_examples_section,
                'keyword args': self._parse_keyword_arguments_section,
                'keyword arguments': self._parse_keyword_arguments_section,
                'methods': self._parse_methods_section,
                'other parameters': self._parse_other_parameters_section,
                'parameters': self._parse_parameters_section,
                'schema': self._parse_schemas_section,
                'schemas': self._parse_schemas_section,
                'map': self._parse_map_section,
                'maps': self._parse_map_section,
                'tag': self._parse_tags_section,
                'tags': self._parse_tags_section,
                'request': self._parse_requests_section,
                'requests': self._parse_requests_section,
                'response': self._parse_responses_section,
                'responses': self._parse_responses_section,
                'return': self._parse_returns_section,
                'returns': self._parse_returns_section,
            }
        self._parse()

    def lines(self) -> List[str]:
        """Return the parsed lines of the docstring in reStructuredText format.
        Returns
        -------
        list(str)
            The lines of the docstring in a list.
        """
        return self._parsed_lines

    def _consume_indented_block(self, indent=1):
        # type: (int) -> List[str]
        lines = []
        line = self._line_iter.peek()
        while(not self._is_section_break() and
              (not line or self._is_indented(line, indent))):
            lines.append(next(self._line_iter))
            line = self._line_iter.peek()
        return lines

    def _consume_contiguous(self):
        # type: () -> List[str]
        lines = []
        while (self._line_iter.has_next() and
               self._line_iter.peek() and
               not self._is_section_header()):
            lines.append(next(self._line_iter))
        return lines

    def _consume_empty(self):
        # type: () -> List[str]
        lines = []
        line = self._line_iter.peek()
        while self._line_iter.has_next() and not line:
            lines.append(next(self._line_iter))
            line = self._line_iter.peek()
        return lines

    def _consume_field(self, parse_type=True, prefer_type=False):
        # type: (bool, bool) -> Tuple[str, str, List[str]]
        line = next(self._line_iter)
        before, colon, after = self._partition_field_on_colon(line)
        _name, _type, _desc = before, '', after  # type: str, str, str
        if parse_type:
            match = _google_typed_arg_regex.match(before)  # type: ignore
            if match:
                _name = match.group(1)
                _type = match.group(2)
        _name = self._escape_args_and_kwargs(_name)
        if prefer_type and not _type:
            _type, _name = _name, _type
        indent = self._get_indent(line) + 1
        _descs = self._dedent(self._consume_indented_block(indent))
        return _name, _type, _descs

    def _consume_fields(self, parse_type=True, prefer_type=False):
        # type: (bool, bool) -> List[Tuple[str, str, List[str]]]
        self._consume_empty()
        fields = []
        while not self._is_section_break():
            _name, _type, _desc = self._consume_field(parse_type, prefer_type)
            if _name or _type or _desc:
                fields.append((_name, _type, _desc,))
        return fields

    def _consume_inline_attribute(self):
        # type: () -> Tuple[str, List[str]]
        line = next(self._line_iter)
        _type, colon, _desc = self._partition_field_on_colon(line)
        if not colon or not _desc:
            _type, _desc = _desc, _type
            _desc += colon
        _descs = [_desc] + self._dedent(self._consume_to_end())
        _descs = self.__class__(_descs, self._config).lines()
        return _type, _descs

    def _join_subsection(self, lines):
        _lines = []
        prev = ""
        for line in lines:
            if line.startswith(" "):
                prev += line
                continue
            else:
                if prev:
                    _lines.append(prev)
                prev = line
        _lines.append(prev)
        retval = {}
        for line in _lines:
            name, response = [x.strip() for x in line.split(":", 1)]
            retval[name] = response
        return retval

    def _parse_responses_section(self, section):
        def _ref_repl(x):
            return re.sub(r'(.+)(:[a-zA-Z0-9]+[\-_+:.])`(.+?)`', r'\3', x)
        lines = self._consume_responses_section()
        lines = [*filter(None, lines)]
        self.responses = {}
        if len(lines) == 1 and lines[0].lower().startswith("see"):
            for split in lines[0].split(" "):
                if re.match(_xref_regex, split):
                    self.responses = {"responses": _ref_repl(lines[0])}
        else:
            self.responses = self._join_subsection(lines)
        return lines

    def _consume_responses_section(self):
        lines = self._dedent(self._consume_to_next_section())
        return lines

    def _consume_returns_section(self):
        # type: () -> List[Tuple[str, str, List[str]]]
        lines = self._dedent(self._consume_to_next_section())
        if len(lines) == 1 and lines[0].lower().startswith("see"):
            for split in lines[0].split(" "):
                if re.match(_xref_regex, split):
                    self.returns = {"returns": _ref_repl(lines[0])}
        else:
            self.returns = [*filter(None, lines)]
            if lines:
                before, colon, after = self._partition_field_on_colon(lines[0])
                _name, _type, _desc = '', '', lines  # type: str, str, List[str]
                if colon:
                    if after:
                        _desc = [after] + lines[1:]
                    else:
                        _desc = lines[1:]
                    _type = before
                return [(_name, _type, _desc,)]
            else:
                return []

    def _consume_usage_section(self):
        # type: () -> List[str]
        lines = self._dedent(self._consume_to_next_section())
        return lines

    def _consume_section_header(self):
        # type: () -> str
        section = next(self._line_iter)
        stripped_section = section.strip(" ").strip(':')
        if stripped_section.lower() in self._sections:
            section = stripped_section
        return section

    def _consume_to_end(self):
        # type: () -> List[str]
        lines = []
        while self._line_iter.has_next():
            lines.append(next(self._line_iter))
        return lines

    def _consume_to_next_section(self):
        # type: () -> List[str]
        self._consume_empty()
        lines = []
        while not self._is_section_break():
            lines.append(next(self._line_iter))
        return lines + self._consume_empty()

    def _dedent(self, lines, full=False):
        # type: (List[str], bool) -> List[str]
        if full:
            return [line.lstrip() for line in lines]
        else:
            min_indent = self._get_min_indent(lines)
            return [line[min_indent:] for line in lines]

    def _escape_args_and_kwargs(self, name):
        # type: (str) -> str
        if name[:2] == '**':
            return r'\*\*' + name[2:]
        elif name[:1] == '*':
            return r'\*' + name[1:]
        else:
            return name

    def _fix_field_desc(self, desc):
        # type: (List[str]) -> List[str]
        if self._is_list(desc):
            desc = [u''] + desc
        elif desc[0].endswith('::'):
            desc_block = desc[1:]
            indent = self._get_indent(desc[0])
            block_indent = self._get_initial_indent(desc_block)
            if block_indent > indent:
                desc = [u''] + desc
            else:
                desc = ['', desc[0]] + self._indent(desc_block, 4)
        return desc

    def _format_admonition(self, admonition, lines):
        # type: (str, List[str]) -> List[str]
        lines = self._strip_empty(lines)
        if len(lines) == 1:
            return ['.. %s:: %s' % (admonition, lines[0].strip()), '']
        elif lines:
            lines = self._indent(self._dedent(lines), 3)
            return [u'.. %s::' % admonition, u''] + lines + [u'']
        else:
            return [u'.. %s::' % admonition, u'']

    def _format_block(self, prefix, lines, padding=None):
        # type: (str, List[str], str) -> List[str]
        if lines:
            if padding is None:
                padding = ' ' * len(prefix)
            result_lines = []
            for i, line in enumerate(lines):
                if i == 0:
                    result_lines.append((prefix + line).rstrip())
                elif line:
                    result_lines.append(padding + line)
                else:
                    result_lines.append('')
            return result_lines
        else:
            return [prefix]

    def _format_docutils_params(self, fields, field_role='param',
                                type_role='type'):
        # type: (List[Tuple[str, str, List[str]]], str, str) -> List[str]  # NOQA
        lines = []
        for _name, _type, _desc in fields:
            _desc = self._strip_empty(_desc)
            if any(_desc):
                _desc = self._fix_field_desc(_desc)
                field = ':%s %s: ' % (field_role, _name)
                lines.extend(self._format_block(field, _desc))
            else:
                lines.append(':%s %s:' % (field_role, _name))

            if _type:
                lines.append(':%s %s: %s' % (type_role, _name, _type))
        return lines + ['']

    def _format_field(self, _name, _type, _desc):
        # type: (str, str, List[str]) -> List[str]
        _desc = self._strip_empty(_desc)
        has_desc = any(_desc)
        separator = has_desc and ' -- ' or ''
        if _name:
            if _type:
                if '`' in _type:
                    field = '**%s** (%s)%s' % (_name, _type, separator)  # type: str
                else:
                    field = '**%s** (*%s*)%s' % (_name, _type, separator)
            else:
                field = '**%s**%s' % (_name, separator)
        elif _type:
            if '`' in _type:
                field = '%s%s' % (_type, separator)
            else:
                field = '*%s*%s' % (_type, separator)
        else:
            field = ''
        if has_desc:
            _desc = self._fix_field_desc(_desc)
            if _desc[0]:
                return [field + _desc[0]] + _desc[1:]
            else:
                return [field] + _desc
        else:
            return [field]

    def _format_fields(self, field_type, fields):
        # type: (str, List[Tuple[str, str, List[str]]]) -> List[str]
        field_type = ':%s:' % field_type.strip()
        padding = ' ' * len(field_type)
        multi = len(fields) > 1
        lines = []  # type: List[str]
        for _name, _type, _desc in fields:
            field = self._format_field(_name, _type, _desc)
            if multi:
                if lines:
                    lines.extend(self._format_block(padding + ' * ', field))
                else:
                    lines.extend(self._format_block(field_type + ' * ', field))
            else:
                lines.extend(self._format_block(field_type + ' ', field))
        if lines and lines[-1]:
            lines.append('')
        return lines

    def _get_current_indent(self, peek_ahead=0):
        # type: (int) -> int
        line = self._line_iter.peek(peek_ahead + 1)[peek_ahead]
        while line != self._line_iter.sentinel:
            if line:
                return self._get_indent(line)
            peek_ahead += 1
            line = self._line_iter.peek(peek_ahead + 1)[peek_ahead]
        return 0

    def _get_indent(self, line):
        # type: (str) -> int
        for i, s in enumerate(line):
            if not s.isspace():
                return i
        return len(line)

    def _get_initial_indent(self, lines):
        # type: (List[str]) -> int
        for line in lines:
            if line:
                return self._get_indent(line)
        return 0

    def _get_min_indent(self, lines):
        # type: (List[str]) -> int
        min_indent = None
        for line in lines:
            if line:
                indent = self._get_indent(line)
                if min_indent is None:
                    min_indent = indent
                elif indent < min_indent:
                    min_indent = indent
        return min_indent or 0

    def _indent(self, lines, n=4):
        # type: (List[str], int) -> List[str]
        return [(' ' * n) + line for line in lines]

    def _is_indented(self, line, indent=1):
        # type: (str, int) -> bool
        for i, s in enumerate(line):
            if i >= indent:
                return True
            elif not s.isspace():
                return False
        return False

    def _is_list(self, lines):
        # type: (List[str]) -> bool
        if not lines:
            return False
        if _bullet_list_regex.match(lines[0]):  # type: ignore
            return True
        if _enumerated_list_regex.match(lines[0]):  # type: ignore
            return True
        if len(lines) < 2 or lines[0].endswith('::'):
            return False
        indent = self._get_indent(lines[0])
        next_indent = indent
        for line in lines[1:]:
            if line:
                next_indent = self._get_indent(line)
                break
        return next_indent > indent

    def _is_section_header(self):
        # type: () -> bool
        section = self._line_iter.peek().lower().strip(" ")
        match = _google_section_regex.match(section)
        if match and section.strip(':') in self._sections:
            header_indent = self._get_indent(section)
            section_indent = self._get_current_indent(peek_ahead=1)
            return section_indent > header_indent
        # elif self._directive_sections:
        #     if _directive_regex.match(section):
        #         for directive_section in self._directive_sections:
        #             if section.startswith(directive_section):
        #                 return True
        return False

    def _is_section_break(self):
        # type: () -> bool
        line = self._line_iter.peek()
        return (not self._line_iter.has_next() or
                self._is_section_header() or
                (self._is_in_section and
                    line and
                    not self._is_indented(line, self._section_indent)))

    # def _load_custom_sections(self):
    #     # type: () -> None

    #     if self._config.napoleon_custom_sections is not None:
    #         for entry in self._config.napoleon_custom_sections:
    #             if isinstance(entry, string_types):
    #                 # if entry is just a label, add to sections list,
    #                 # using generic section logic.
    #                 self._sections[entry.lower()] = self._parse_custom_generic_section
    #             else:
    #                 # otherwise, assume entry is container;
    #                 # [0] is new section, [1] is the section to alias.
    #                 # in the case of key mismatch, just handle as generic section.
    #                 self._sections[entry[0].lower()] = \
    #                     self._sections.get(entry[1].lower(),
    #                                        self._parse_custom_generic_section)

    def _parse(self):
        # type: () -> None
        self._parsed_lines = self._consume_empty()
        if self._name and (self._what == 'attribute' or self._what == 'data'):
            # Implicit stop using StopIteration no longer allowed in
            # Python 3.7; see PEP 479
            res = []  # type: List[str]
            try:
                res = self._parse_attribute_docstring()
            except StopIteration:
                pass
            self._parsed_lines.extend(res)
            return
        while self._line_iter.has_next():
            if self._is_section_header():
                try:
                    section = self._consume_section_header()
                    self._is_in_section = True
                    self._section_indent = self._get_current_indent()
                    if _directive_regex.match(section):  # type: ignore
                        lines = [section] + self._consume_to_next_section()
                    else:
                        lines = self._sections[section.lower()](section)
                finally:
                    self._is_in_section = False
                    self._section_indent = 0
            else:
                if not self._parsed_lines:
                    lines = self._consume_contiguous() + self._consume_empty()
                    self.description = " ".join([line.lstrip() for line in lines])
                else:
                    lines = self._consume_to_next_section()
            self._parsed_lines.extend(lines)

    def _parse_attribute_docstring(self):
        # type: () -> List[str]
        _type, _desc = self._consume_inline_attribute()
        lines = self._format_field('', '', _desc)
        if _type:
            lines.extend(['', ':type: %s' % _type])
        return lines

    def _parse_attributes_section(self, section):
        # type: (str) -> List[str]
        lines = []
        for _name, _type, _desc in self._consume_fields():
            lines.extend(['.. attribute:: ' + _name, ''])
            fields = self._format_field('', '', _desc)
            lines.extend(self._indent(fields, 3))
            if _type:
                lines.append('')
                lines.extend(self._indent([':type: %s' % _type], 3))
            lines.append('')
        return lines

    def _parse_examples_section(self, section) -> Tuple[str, List[str]]:
        labels = {
            'example': _('Example'),
            'examples': _('Examples'),
        }  # type: Dict[str, str]
        label = labels.get(section.lower(), section)
        lines = self._strip_empty(self._consume_to_next_section())
        lines = self._dedent(lines)
        self.examples = section, lines
        return section, lines

    def _parse_custom_generic_section(self, section):
        # for now, no admonition for simple custom sections
        return self._parse_generic_section(section, False)

    def _parse_usage_section(self, section):
        # type: (str) -> List[str]
        header = ['.. rubric:: Usage:', '']  # type: List[str]
        block = ['.. code-block:: python', '']  # type: List[str]
        lines = self._consume_usage_section()
        lines = self._indent(lines, 3)
        return header + block + lines + ['']

    def _parse_generic_section(self, section, use_admonition):
        # type: (str, bool) -> List[str]
        lines = self._strip_empty(self._consume_to_next_section())
        lines = self._dedent(lines)
        if use_admonition:
            header = '.. admonition:: %s' % section  # type: str
            lines = self._indent(lines, 3)
        else:
            header = '.. rubric:: %s' % section
        if lines:
            return [header, ''] + lines + ['']
        else:
            return [header, '']

    def _parse_keyword_arguments_section(self, section):
        # type: (str) -> List[str]
        fields = self._consume_fields()
        return self._format_docutils_params(
            fields,
            field_role="keyword",
            type_role="kwtype")

    def _parse_methods_section(self, section):
        # type: (str) -> List[str]
        lines = []  # type: List[str]
        for _name, _type, _desc in self._consume_fields(parse_type=False):
            lines.append('.. method:: %s' % _name)
            if _desc:
                lines.extend([u''] + self._indent(_desc, 3))
            lines.append('')
        return lines

    def _parse_other_parameters_section(self, section):
        # type: (str) -> List[str]
        fields = self._consume_fields()
        self.parameters.extend(fields)
        return self._format_fields(_('Other Parameters'), fields)

    def _parse_parameters_section(self, section):
        # type: (str) -> List[str]
        fields = self._consume_fields()
        self.parameters = fields
        return self._format_docutils_params(fields)

    def _parse_tags_section(self, section: str) -> Tuple[str, List[str]]:
        lines = self._strip_empty(self._consume_to_next_section())
        lines = self._dedent(lines)
        self.tags = ",".join(lines).replace(",,", ",")
        return section, lines

    def _parse_map_section(self, section: str) -> Tuple[str, List[str]]:
        lines = self._strip_empty(self._consume_to_next_section())
        lines = self._dedent(lines)
        self.map = lines[0]
        return section, lines

    def _parse_schemas_section(self, section: str) -> Tuple[str, List[str]]:
        lines = self._strip_empty(self._consume_to_next_section())
        lines = self._dedent(lines)
        self.schemas = section, lines
        return section, lines

    def _parse_requests_section(self, section: str) -> Tuple[str, List[str]]:
        lines = self._strip_empty(self._consume_to_next_section())
        lines = self._dedent(lines)
        self.requests = section, lines
        return section, lines

    def _parse_returns_section(self, section):
        # type: (str) -> List[str]
        fields = self._consume_returns_section()
        multi = len(fields) > 1
        use_rtype = True
        lines = []  # type: List[str]
        for _name, _type, _desc in fields:
            if use_rtype:
                field = self._format_field(_name, '', _desc)
            else:
                field = self._format_field(_name, _type, _desc)
            if multi:
                if lines:
                    lines.extend(self._format_block('          * ', field))
                else:
                    lines.extend(self._format_block(':returns: * ', field))
            else:
                lines.extend(self._format_block(':returns: ', field))
                if _type and use_rtype:
                    lines.extend([':rtype: %s' % _type, ''])
        if lines and lines[-1]:
            lines.append('')
        return lines

    def _partition_field_on_colon(self, line):
        # type: (str) -> Tuple[str, str, str]
        before_colon = []
        after_colon = []
        colon = ''
        found_colon = False
        for i, source in enumerate(_xref_regex.split(line)):  # type: ignore
            if found_colon:
                after_colon.append(source)
            else:
                m = _single_colon_regex.search(source)
                if (i % 2) == 0 and m:
                    found_colon = True
                    colon = source[m.start(): m.end()]
                    before_colon.append(source[:m.start()])
                    after_colon.append(source[m.end():])
                else:
                    before_colon.append(source)
        return ("".join(before_colon).strip(),
                colon,
                "".join(after_colon).strip())

    def _qualify_name(self, attr_name, klass):
        # type: (str, Type) -> str
        if klass and '.' not in attr_name:
            if attr_name.startswith('~'):
                attr_name = attr_name[1:]
            try:
                q = klass.__qualname__
            except AttributeError:
                q = klass.__name__
            return '~%s.%s' % (q, attr_name)
        return attr_name

    def _strip_empty(self, lines):
        # type: (List[str]) -> List[str]
        if lines:
            start = -1
            for i, line in enumerate(lines):
                if line:
                    start = i
                    break
            if start == -1:
                lines = []
            end = -1
            for i in reversed(range(len(lines))):
                line = lines[i]
                if line:
                    end = i
                    break
            if start > 0 or end + 1 < len(lines):
                lines = lines[start:end + 1]
        return lines
